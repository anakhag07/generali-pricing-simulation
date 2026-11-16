"""Checkpointed polynomial-residual mode of the existing wild-envelope study."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
from scipy.special import expit

from experiments.launch import LaunchPlan
from experiments.provenance import (array_sha256, file_record, read_json,
                                    save_arrays_atomic, write_json_atomic)
from experiments.seeds import derive_seed
from experiments.slurm import CPU_PROFILE
from objective.base import Objective, Policy
from optimization.solvers import run_first_order_minimize, run_stein_difference_minimize
from .feature_wild import WildFeatureDesign
from .polynomial import (AdditivePolynomial, PolynomialFeatureBand, polynomial_features,
                         fit_polynomial_band, coverage_events, certify_policy_gap,
                         _unavailable_coverage)


def load_polynomial_manifest(path, payload):
    """Validate the explicit first-stage scientific contract before sampling."""
    from .wild_study import WildEnvelopeManifest
    p = payload
    if p['study_mode'] != 'polynomial_residual' or p['band_target'] != 'residual':
        raise ValueError('Require polynomial_residual mode and residual target.')
    name = p['name']
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
        raise ValueError('Require a safe result folder name.')
    degrees = p['feature_degrees']
    if degrees != sorted(set(degrees)) or any(type(k) is not int or k < 1 or k > 6 for k in degrees):
        raise ValueError('Require unique increasing polynomial degrees in 1..6.')
    if type(p['n']) is not int or p['n'] <= max(degrees)+4:
        raise ValueError('Require n greater than every feature dimension.')
    if p['design']['context_dim'] != 3 or p['design']['action_variance'] <= 0:
        raise ValueError('Require three contexts and positive action variance.')
    if 'models' in p:
        from .structured_residual import validate_structured
        validate_structured(p)
    elif p['target'] != {'contexts': 'calibration', 'actions': 'real_line'}:
        raise ValueError('Require all calibration contexts and every real action.')
    if p['bootstrap']['multipliers'] != 'rademacher' or p['bootstrap']['residuals'] != 'raw':
        raise ValueError('Require raw-residual Rademacher bootstrap.')
    if p['bootstrap']['calibration'] != 'coefficient_ellipsoid' or p['transfer'] != 'asymmetric':
        raise ValueError('Require coefficient ellipsoid and asymmetric transfer.')
    if type(p['bootstrap']['B']) is not int or p['bootstrap']['B'] < 2 or not 0 < p['bootstrap']['delta'] < 1:
        raise ValueError('Invalid bootstrap count or miscoverage level.')
    if p['noise']['type'] != 'squared_exponential_gp_restriction':
        raise ValueError('Require the exact squared-exponential finite restriction.')
    if not 0 < p['noise']['sigma_min'] < p['noise']['sigma_max'] or min(p['noise']['lengthscale_x'], p['noise']['lengthscale_a']) <= 0:
        raise ValueError('Invalid noise scale/lengthscale.')
    for key in ('truth', 'residual'):
        model = p[key]
        if len(model['context']) != 3 or not model['action'] or not np.isfinite([*model['context'], *model['action']]).all():
            raise ValueError('Require explicit finite additive function coefficients.')
    for key in ('repetitions', 'repetitions_per_task'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError('Require positive repetition counts.')
    for key in ('contexts', 'actions', 'profile', 'observations', 'method', 'optimizer'):
        if type(p['seeds'][key]) is not int or p['seeds'][key] < 0:
            raise ValueError('Require separate named nonnegative integer seed roots.')
    o = p['optimizer']
    if o['method'] != 'stein_difference' or o['perturbation_space'] != 'u':
        raise ValueError('Require native action-space Stein differences.')
    if o['step_rule'] not in ('constant', 'armijo', 'l-bfgs-b') or not o['starts']:
        raise ValueError('Invalid repository step rule or starts.')
    if min(o['t_steps'], o['n_grad_samples'], o['smoothing_sd'], o['step_size'], o['gradient_tolerance']) <= 0:
        raise ValueError('Require positive optimizer budget and scales.')
    launch = p['launch']
    return WildEnvelopeManifest(p, Path(path).resolve(), SimpleNamespace(
        mode=launch['mode'], array=launch['array'], array_max_parallel=launch['array_max_parallel']))


def prepare_environment(p):
    """Draw the fixed design and exact finite GP restriction using separate streams."""
    n, seeds, design, noise = p['n'], p['seeds'], p['design'], p['noise']
    X = np.random.default_rng(seeds['contexts']).normal(size=(n, 3))
    action_z = np.random.default_rng(seeds['actions']).normal(size=n)
    actions = design['action_mean']+np.sqrt(design['action_variance'])*action_z
    distance = np.sum((X[:, None, :]-X[None, :, :])**2, axis=2)/noise['lengthscale_x']**2
    distance += (actions[:, None]-actions[None, :])**2/noise['lengthscale_a']**2
    covariance = np.exp(-distance/2)
    cholesky = np.linalg.cholesky(covariance)  # No silent nugget or kernel replacement.
    gp_z = np.random.default_rng(seeds['profile']).normal(size=n)
    profile = cholesky @ gp_z
    sigma = noise['sigma_min']+(noise['sigma_max']-noise['sigma_min'])*expit(noise['amplitude']*profile)
    return dict(X=X, actions=actions, action_innovations=action_z, gp_covariance=covariance,
                gp_cholesky=cholesky, gp_innovations=gp_z, gp_values=profile, sigma=sigma)


def _runtime_sources():
    root = Path(__file__).resolve().parents[3]
    names = ['MATH.md', 'src/experiments/envelopes/polynomial.py',
             'src/experiments/envelopes/mlp_head.py', 'src/experiments/envelopes/structured_residual.py',
             'src/experiments/envelopes/polynomial_study.py', 'src/experiments/envelopes/feature_wild.py',
             'src/experiments/envelopes/wild.py', 'src/experiments/envelopes/quadratic.py',
             'src/experiments/offline_bootstrap_math.py', 'src/experiments/seeds/streams.py',
             'src/experiments/provenance.py', 'src/objective/base.py', 'src/objective/utils.py']
    names += [str(p.relative_to(root)) for p in (root/'src/optimization').rglob('*.py')]
    return root, [root/name for name in sorted(names)]


def prepare(manifest, runs_root):
    """Create or verify immutable environment, source snapshot and experiment contract."""
    p = manifest.payload
    project = Path(runs_root)/manifest.name
    project.mkdir(parents=True, exist_ok=True)
    root, sources = _runtime_sources()
    payload = {'manifest': p, 'sources': [file_record(s) for s in sources],
               'packages': {name: version(name) for name in ('numpy', 'scipy', 'sympy')}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = project/'contract.json'
    if path.exists():
        saved = read_json(path)
        if saved['sha256'] != digest or saved['environment'] != file_record(project/'inputs.npz'):
            raise ValueError('Completed input/source contract differs; use a new result folder.')
        with np.load(project/'inputs.npz', allow_pickle=False) as data:
            snapshot = dict(data)
    else:
        snapshot = prepare_environment(p)
        save_arrays_atomic(project/'inputs.npz', **snapshot)
        copies = []
        for source in sources:
            target = project/'source'/source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copies.append(file_record(target))
        write_json_atomic(project/'config.json', p)
        saved = {'sha256': digest, 'contract': payload, 'environment': file_record(project/'inputs.npz'),
                 'source_snapshot': copies, 'source_worktree': str(root),
                 'repository_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()}
        write_json_atomic(path, saved)
        guide = 'synthetic_residual_extensions.md' if 'models' in p else 'synthetic_polynomial_residual.md'
        if p.get('reporting', {}).get('fit_diagnostics'):
            guide = 'synthetic_mlp_capacity.md'
        shutil.copyfile(root/'docs'/guide, project/'EXPERIMENT.md')
    designs = {} if 'models' in p else {k: WildFeatureDesign(polynomial_features(snapshot['X'], snapshot['actions'], k))
                                       for k in p['feature_degrees']}
    return project, digest, snapshot, designs


def run_repetition(p, index, project, digest, snapshot, designs):
    """Fit paired feature degrees and save actual all-real coverage decisions."""
    stem = project/'repetitions'/f'rep-{index:03d}'
    if stem.with_suffix('.json').exists():
        record = read_json(stem.with_suffix('.json'))
        if record['contract_sha256'] != digest or record['arrays'] != file_record(stem.with_suffix('.npz')):
            raise ValueError('Corrupt or incompatible saved repetition.')
        return record
    seeds = {key: derive_seed(p['seeds'][key], f'repetition:{index}') for key in ('observations', 'method')}
    innovations = np.random.default_rng(seeds['observations']).normal(size=p['n'])
    signs = 2*np.random.default_rng(seeds['method']).integers(0, 2, size=(p['bootstrap']['B'], p['n']), dtype=np.int8)-1
    truth, residual = _models(p)
    X, actions = snapshot['X'], snapshot['actions']
    observations = truth(X, actions)+snapshot['sigma']*innovations
    responses = observations-(truth(X, actions)-residual(X, actions))
    arrays = {'innovations': innovations, 'multipliers': signs, 'observations': observations,
              'residual_observations': responses}
    events, diagnostics = {}, {}
    if 'models' in p:
        from .structured_residual import fit_structured_models
        events, diagnostics = fit_structured_models(p, index, project, digest, snapshot, responses, signs, arrays)
    for degree in ([] if 'models' in p else p['feature_degrees']):
        band, state = fit_polynomial_band(X, actions, responses, degree, signs,
                                         p['bootstrap']['delta'], design=designs[degree])
        try:
            band.evaluate(actions)
            event = coverage_events(band, residual)
        except FloatingPointError as error:
            event = _unavailable_coverage(error)
            state['band_failure'] = np.array(str(error))
        events[str(degree)] = event
        arrays.update({f'degree{degree}_{key}': value for key, value in state.items()})
    save_arrays_atomic(stem.with_suffix('.npz'), **arrays)
    record = {'index': index, 'contract_sha256': digest, 'seeds': seeds,
              'multiplier_sha256': array_sha256(signs), 'events': events,
              'arrays': file_record(stem.with_suffix('.npz'))}
    if 'models' in p:
        record['model_diagnostics'] = diagnostics
    write_json_atomic(stem.with_suffix('.json'), record)
    print(f"Repetition {index+1}/{p['repetitions']}: transferred "
          f"{[int(event['transferred']) if event['transferred'] is not None else 'unavailable' for event in events.values()]}", flush=True)
    return record


def restore_band(snapshot, state, degree):
    """Restore exact numeric query state without fitting or sampling."""
    if f'degree{degree}_band_failure' in state:
        raise ValueError(f"No calibration band: {state[f'degree{degree}_band_failure']}")
    return PolynomialFeatureBand(snapshot['X'], degree, state[f'degree{degree}_beta'],
                                 state[f'degree{degree}_inverse'], state[f'degree{degree}_critical'])


class ContextActions(Policy):
    """One free action per context in the fixed ordered full batch."""

    def value(self, theta, x_batch):
        if np.shape(theta) != (len(x_batch),):
            raise ValueError('Identity policy requires the full ordered context batch.')
        return np.asarray(theta)

    def weighted_grad(self, theta, x_batch, weights):
        self.value(theta, x_batch)
        return np.asarray(weights)


class PolynomialLCBObjective(Objective):
    """Negative asymmetric LCB exposed only through exact function-value queries."""

    policy = ContextActions()

    def __init__(self, band, truth, residual):
        self.band, self.truth, self.residual = band, truth, residual
        self.query_count = 0
        self.query_min = np.full(len(band.contexts), np.inf)
        self.query_max = np.full(len(band.contexts), -np.inf)

    def _value_batch_many(self, x_batch, actions):
        if not np.array_equal(x_batch, self.band.contexts):
            raise ValueError('Full ordered context queries are required.')
        a = np.asarray(actions)
        self.query_count += a.size
        self.query_min = np.minimum(self.query_min, a.min(axis=0))
        self.query_max = np.maximum(self.query_max, a.max(axis=0))
        _, _, lower, _ = self.band.evaluate(a)
        return -self.truth(x_batch, a)+self.residual(x_batch, a)+lower

    def _value_batch(self, x_batch, actions):
        return self._value_batch_many(x_batch, np.asarray(actions)[None, :])[0]

    def value(self, theta, x_batch):
        return float(np.mean(self._value_batch(x_batch, theta)))

    def grad(self, theta, x_batch):
        raise AssertionError('The LCB must use Stein function-value queries.')


class _TruthObjective(Objective):
    policy = ContextActions()

    def __init__(self, truth):
        self.truth = truth

    def value(self, theta, x_batch):
        return -float(np.mean(self.truth(x_batch, theta)))

    def grad(self, theta, x_batch):
        return -np.polynomial.polynomial.polyval(theta, np.polynomial.polynomial.polyder(self.truth.action))/len(x_batch)


def optimize_headline(p, project, digest, snapshot):
    """Run/replay repository endpoints, preserving numerical failures as unavailable policies."""
    output = project/'policies'
    output.mkdir(exist_ok=True)
    truth, residual = _models(p)
    if 'models' in p:
        from .structured_residual import evaluation_rows
        X = snapshot['X'][evaluation_rows(p)]
        entries = [(m['name'], m['name']) for m in p['models']]
    else:
        X = snapshot['X']
        entries = [(k, f'degree-{k}') for k in p['feature_degrees']]
    n = len(X)
    reference_path = output/'truth.json'
    if reference_path.exists():
        reference = _verified_record(reference_path, digest)
    else:
        theta, trace = run_first_order_minimize(np.zeros(n), X, _TruthObjective(truth),
            t_steps=100, n_grad_samples=1, sigma=0., grad_norm_tol=1e-12, ftol=1e-14)
        if not trace.optimizer_success or not np.allclose(theta, .5, atol=1e-8, rtol=0):
            raise RuntimeError('Repository truth solve failed the supplied quadratic reference check.')
        save_arrays_atomic(output/'truth.npz', actions=theta, theta_values=np.asarray(trace.theta_values),
                           losses=trace.objective_values)
        reference = {'contract_sha256': digest, 'solver': 'optimization.solvers.run_first_order_minimize',
                     'success': bool(trace.optimizer_success), 'max_error_from_supplied_truth': float(np.max(abs(theta-.5))),
                     'arrays': file_record(output/'truth.npz')}
        write_json_atomic(reference_path, reference)
    repetition = _verified_record(project/'repetitions/rep-000.json', digest)
    with np.load(project/'repetitions/rep-000.npz', allow_pickle=False) as saved:
        state = dict(saved)
    for degree, label in entries:
        path = output/f'{label}.json'
        if path.exists():
            _verified_record(path, digest)
            continue
        event = repetition['events'][str(degree)]
        if event['transferred'] is None:
            _save_unavailable_policy(path, digest, repetition, 'coverage',
                                     event.get('failure', event['certificate']))
            continue
        try:
            _optimize_headline_band(p, project, digest, snapshot, state, degree, label)
        except FloatingPointError as error:
            _save_unavailable_policy(path, digest, repetition, 'policy_evaluation', str(error))


def _save_unavailable_policy(path, digest, repetition, stage, reason):
    write_json_atomic(path, {'contract_sha256': digest, 'status': 'unavailable',
        'stage': stage, 'failure': reason, 'arrays': repetition['arrays']})
    print(f"{path.stem}: policy unavailable ({stage}): {reason}", flush=True)


def _optimize_headline_band(p, project, digest, snapshot, state, degree, label):
    from .structured_residual import evaluation_rows, restore_structured_band
    X = snapshot['X'][evaluation_rows(p)] if 'models' in p else snapshot['X']
    n, settings = len(X), p['optimizer']
    truth, residual = _models(p)
    output = project/'policies'
    path = output/f'{label}.json'
    band = (restore_structured_band(p, snapshot, state, degree) if 'models' in p
            else restore_band(snapshot, state, degree))
    attempts, vectors, losses = [], [], []
    for j, start in enumerate(settings['starts']):
        attempt_path = output/f'{label}-start-{j}.json'
        if attempt_path.exists():
            info = _verified_record(attempt_path, digest)
            with np.load(info['arrays']['path'], allow_pickle=False) as saved:
                theta, final_losses = saved['actions'], saved['final_losses']
        else:
            seed = derive_seed(p['seeds']['optimizer'], f'stein-difference:start:{j}')
            batch_seed = derive_seed(p['seeds']['optimizer'], f'batch:start:{j}')
            objective = PolynomialLCBObjective(band, truth, residual)
            theta, trace = run_stein_difference_minimize(np.full(n, start), X, objective,
                np.random.default_rng(seed), t_steps=settings['t_steps'],
                n_grad_samples=settings['n_grad_samples'], sigma=settings['smoothing_sd'],
                perturbation_space='u', algorithm=settings['step_rule'],
                step_size=n*settings['step_size'], grad_norm_tol=settings['gradient_tolerance']/np.sqrt(n),
                ftol=settings.get('ftol'),
                gradient_rng=np.random.default_rng(seed), batch_rng=np.random.default_rng(batch_seed))
            final_losses = objective._value_batch(X, theta)
            if not np.isfinite(theta).all() or not np.isfinite(final_losses).all():
                raise FloatingPointError('Nonfinite repository endpoint; headline remains incomplete.')
            arrays_path = attempt_path.with_suffix('.npz')
            save_arrays_atomic(arrays_path, actions=theta, theta_values=np.asarray(trace.theta_values),
                losses=trace.objective_values, gradient_norms=trace.theta_grad_norms,
                final_losses=final_losses, query_min=objective.query_min, query_max=objective.query_max)
            info = {'contract_sha256': digest, 'start': start, 'seed': seed, 'batch_seed': batch_seed,
                    'success': bool(trace.optimizer_success), 'message': trace.optimizer_message,
                    'steps': int(trace.steps[-1]), 'query_count': objective.query_count,
                    'arrays': file_record(arrays_path)}
            write_json_atomic(attempt_path, info)
        attempts.append(info)
        vectors.append(theta)
        losses.append(final_losses)
        print(f"Degree {degree}, start {start}: success={info['success']}, steps={info['steps']}", flush=True)
    # Only actual repository endpoints compete; no action/evaluation grid is searched.
    selected = np.argmin(np.asarray(losses), axis=0)
    actions = np.asarray(vectors)[selected, np.arange(n)]
    polynomial_certificate = p['target']['actions'] == 'real_line'
    certificates = (certify_policy_gap(band, truth, residual, actions, settings['global_gap_tolerance'])
                    if polynomial_certificate else np.full(n, -1, dtype=np.int8))
    selected_success = np.array([attempts[int(j)]['success'] for j in selected])
    save_arrays_atomic(path.with_suffix('.npz'), actions=actions, selected_start=selected,
                       all_start_actions=np.asarray(vectors), gap_certified=certificates,
                       selected_solver_success=selected_success)
    record = {'contract_sha256': digest, 'degree': degree,
              'solver': 'optimization.solvers.run_stein_difference_minimize', 'attempts': attempts,
              'selected_success_count': int(selected_success.sum()),
              'global_gap_tolerance': settings['global_gap_tolerance'],
              'global_gap_certified_count': int(certificates.sum()) if polynomial_certificate else None,
              'arrays': file_record(path.with_suffix('.npz'))}
    write_json_atomic(path, record)
    print(f"Model {degree}: global gap certificates {record['global_gap_certified_count']}/{n}", flush=True)


def _verified_record(path, digest):
    record = read_json(path)
    if record['contract_sha256'] != digest or record['arrays'] != file_record(record['arrays']['path']):
        raise ValueError('Saved policy differs from its input/source or checksum contract.')
    return record


def _models(p):
    from .structured_residual import explicit_model
    return tuple(explicit_model(p[key]) for key in ('truth', 'residual'))


def _collect_with_band_status(project, collector):
    project = Path(project)
    contract = read_json(project/'contract.json')
    p, digest = contract['contract']['manifest'], contract['sha256']
    labels = ([m['name'] for m in p['models']] if 'models' in p
              else [f'degree-{k}' for k in p['feature_degrees']])
    unavailable = []
    for label in labels:
        path = project/'policies'/f'{label}.json'
        if path.exists():
            policy = _verified_record(path, digest)
            if policy.get('status') == 'unavailable':
                unavailable.append({'model': label, 'failure': policy['failure'],
                                    'stage': policy['stage'], 'record': file_record(path)})
    if unavailable:
        stage, reason = 'headline_policy', 'A complete policy comparison requires every headline band and policy.'
    else:
        try:
            return collector(project)
        except FloatingPointError as error:
            stage, reason = 'report_evaluation', str(error)
    record = {'status': 'unavailable', 'stage': stage, 'failure': reason,
              'contract_sha256': digest, 'contract': file_record(project/'contract.json'),
              'unavailable_policies': unavailable}
    write_json_atomic(project/'report-unavailable.json', record)
    print(f'Report unavailable ({stage}): {reason}', flush=True)
    return record


def build_polynomial_launch_plan(manifest, *, runs_root=None, force=False):
    """Reuse the family launcher, preserving completed repetitions and solver endpoints."""
    if force:
        raise ValueError('Polynomial runs preserve completed artifacts; choose a fresh folder for revisions.')
    p = manifest.payload
    count = int(np.ceil(p['repetitions']/p['repetitions_per_task']))
    def run_task(index, context):
        if not 0 <= index < count:
            raise IndexError(index)
        project, digest, snapshot, designs = prepare(manifest, context.runs_root)
        indices = range(index*p['repetitions_per_task'], min((index+1)*p['repetitions_per_task'], p['repetitions']))
        for repetition in indices:
            run_repetition(p, repetition, project, digest, snapshot, designs)
            if repetition == 0:
                optimize_headline(p, project, digest, snapshot)
        return {'repetitions': list(indices)}
    def collect(context):
        if 'models' in p:
            from .structured_reporting import collect_structured
            return collect_structured(Path(context.runs_root)/manifest.name)
        from .polynomial_reporting import collect_polynomial
        return collect_polynomial(Path(context.runs_root)/manifest.name)
    def run_all(context):
        for index in range(count):
            run_task(index, context)
        return collect(context)
    return LaunchPlan(name=manifest.name, task_count=count, requires_jax=False,
                      run_task=run_task, run_all=run_all, collect=collect, runs_root=runs_root,
                      default_launch=manifest.launch.mode, default_array=manifest.launch.array == 'repetition',
                      slurm_profile=replace(CPU_PROFILE, cpus_per_task=1, memory='4G'))
