"""Parity, replay, optimizer composition and saved-contract integration."""
from pathlib import Path
import json
import numpy as np
import pytest
from experiments.envelopes.core import CalibrationData, TargetDomain
from experiments.envelopes.linear import LinearModel, QuadraticModel, fit_linear_transfer
from experiments.envelopes.wild import WildLinearDesign, residual_radii
from experiments.envelopes.fitting import fit_band, save_band, load_band
from experiments.envelopes.support import fit_support_band
from experiments.manifest import parse_experiment_manifest, run_manifest_variant, variant_complete, collect_manifest_outputs
from experiments.configs import get_config


def observed():
    actions = np.array([-2., -1.3, -.7, -.1, .4, .8, 1.2, 1.8])
    X = np.empty((len(actions), 0))
    y = 1 + actions + actions**2 + np.array([.2, -.1, .3, -.4, .1, .2, -.2, .4])
    return CalibrationData(X, actions, y), QuadraticModel([0., 0., 1.])


@pytest.mark.parametrize('method', ['residual_ols', 'residual_wild'])
def test_residual_fit_parity_and_replay_without_refitting(tmp_path, monkeypatch, method):
    data, predictor = observed()
    settings = dict(method=method, delta=.2)
    if method == "residual_ols":
        settings["sigma_bound"] = .4
    multipliers = np.array([[1,-1]*4, [-1,1]*4, [1]*8, [-1]*8])
    band = fit_band(data, predictor, TargetDomain(data.X), settings, multipliers=multipliers)
    design = WildLinearDesign(data.X, data.actions, action_degree=2)
    beta = design.fit(data.y-predictor(data.X, data.actions))
    if method == 'residual_wild':
        expected_beta, errors = design.bootstrap_errors(data.y, multipliers, predictor=predictor(data.X, data.actions))
        np.testing.assert_array_equal(beta, expected_beta)
        critical = np.sort(design.statistic_bounds(errors)[:,1])[3]
    else:
        critical = .4*np.sqrt(3/.2)
    phi = design.features(data.X, data.actions)
    expected = residual_radii(phi@beta, critical*np.sqrt(np.einsum('ij,jk,ik->i', phi, design.covariance, phi)))
    values = band.evaluate(data.X, data.actions)
    assert values.radius is None
    np.testing.assert_allclose(values.radius_lower, expected[0])
    np.testing.assert_allclose(values.radius_upper, expected[1])
    for endpoint in ('lower', 'upper', 'center'):
        plus = getattr(band.evaluate(data.X, data.actions+1e-6), endpoint)
        minus = getattr(band.evaluate(data.X, data.actions-1e-6), endpoint)
        np.testing.assert_allclose(band.derivative(data.X, data.actions, endpoint), (plus-minus)/2e-6, atol=1e-6)
    save_band(band, tmp_path/'band.json')
    monkeypatch.setattr(np.linalg, 'qr', lambda *a, **k: pytest.fail('Replay must not refit'))
    restored = load_band(tmp_path/'band.json')
    np.testing.assert_array_equal(restored.evaluate(data.X, data.actions).lower, values.lower)
    with open(tmp_path/'band.npz', 'ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        load_band(tmp_path/'band.json')


def test_ols_transfer_parity():
    raw, _ = observed()
    data = CalibrationData(raw.X, raw.actions, raw.y)
    predictor = LinearModel([.1, .4])
    target = TargetDomain(data.X)
    settings = dict(method='ols_transfer', delta=.1, sigma_bound=.4)
    original = fit_linear_transfer(data, predictor, target, settings, None).evaluate(data.X, data.actions)
    fitted = fit_band(data, predictor, target, settings).evaluate(data.X, data.actions)
    np.testing.assert_array_equal(original.center, fitted.center)
    np.testing.assert_allclose(original.radius, fitted.radius)


def support_band():
    return fit_support_band(np.array([[0.], [1.]]), [-1., 0., 1.],
        np.array([[2., 3., 1.], [1., 4., 2.]]),
        dict(construction='normalized_local', scale=.2, quantity='profit'),
        support=np.array([[1.,2.,1.], [2.,3.,1.]]),
        acceptance_grid=np.array([[.8,.6,.4],[.9,.7,.5]]), row_indices=np.array([4,8]))


def test_support_bounds_row_alignment_gradient_and_controls():
    band = support_band()
    cfg = get_config('band_policy', dict(fitted_band=band, band_artifact={}, policy_kind='softmax',
        endpoint='lower', sense='maximize', step_rule='l-bfgs-b', enabled_estimators=['first_order'],
        softmax_action_bounds=(-1.,1.), perturbation_space='theta', acceptance_floor=.65,
        acceptance_penalty_weight=2.))
    theta = np.array([.3, .2])
    objective = cfg.objective
    eps = 1e-6
    finite = [(objective.value(theta+eps*v, cfg.x_fixed)-objective.value(theta-eps*v, cfg.x_fixed))/(2*eps) for v in np.eye(len(theta))]
    np.testing.assert_allclose(objective.grad(theta, cfg.x_fixed), finite, atol=1e-7)
    values = band.evaluate(np.array([[1.],[0.]]), np.array([.5,-.5]))
    np.testing.assert_allclose(values.center, [3.,2.5])
    assert band.state['row_indices'].dtype.kind == 'i'
    with pytest.raises(ValueError, match='outside'):
        band.evaluate(np.array([[0.]]), np.array([1.1]))
    with pytest.raises(ValueError, match='saved customer'):
        band.evaluate(np.array([[2.]]), np.array([0.]))


def payload(tmp_path, task='band'):
    data, _ = observed()
    np.savez(tmp_path/'observations.npz', X=data.X, actions=data.actions, y=data.y)
    result = dict(name='band-workflow-test', task=task,
        setting=dict(source='observations', path=str(tmp_path/'observations.npz'), quantity='synthetic response',
            provenance='deterministic test fixture', predictor=dict(type='quadratic', coefficients=[0.,0.,1.])),
        band=dict(method='residual_ols', delta=.2, sigma_bound=.4), launch=dict(mode='local', array='variant'))
    if task == 'optimize_with_band':
        result.update(objective=dict(preset='band_policy', overrides=dict(policy_kind='constant', endpoint='upper', sense='minimize')),
            objective_modifications=[], optimizer=dict(step_rule='l-bfgs-b', enabled_estimators=['first_order', 'finite_difference'],
                perturbation_space='theta', t_steps=3, n_grad_samples=2, theta0=[.3]),
            seeds=dict(run_seeds=[7], anchor_seed=7, vary=['theta']))
    return result


@pytest.mark.parametrize('task', ['band','optimize_with_band'])
def test_manifest_end_to_end_and_incompatible_resume(tmp_path, monkeypatch, task):
    p = payload(tmp_path, task)
    manifest = parse_experiment_manifest(p)
    variant = manifest.variants[0]
    assert run_manifest_variant(manifest, 0, runs_root=tmp_path)['n_runs'] == 1
    assert variant_complete(manifest, variant, runs_root=tmp_path)
    monkeypatch.setattr('experiments.band_workflow.execute_band_variant', lambda *a, **k: pytest.fail('must not rerun'))
    assert run_manifest_variant(manifest, 0, runs_root=tmp_path)['skipped']
    collect_manifest_outputs(manifest, runs_root=tmp_path)
    folder = manifest.variant_dir(variant, tmp_path)
    assert (folder/'band-summary.json').is_file()
    if task != 'band':
        assert len(list(folder.glob('seeds/seed-7/band-policy-*.npz'))) == 2
    p['band']['delta'] = .1
    changed = parse_experiment_manifest(p)
    with pytest.raises(ValueError, match='contract changed'):
        run_manifest_variant(changed, 0, runs_root=tmp_path)
    with pytest.raises(ValueError, match='differs'):
        collect_manifest_outputs(changed, runs_root=tmp_path)


def test_band_only_requires_no_optimizer_and_saved_band_loads(tmp_path):
    p = payload(tmp_path)
    manifest = parse_experiment_manifest(p)
    run_manifest_variant(manifest, 0, runs_root=tmp_path)
    p.pop('setting')
    p['name'] = 'band-replay-test'
    p['band'] = {'artifact': str(manifest.variant_dir(manifest.variants[0], tmp_path)/'band.json')}
    replay = parse_experiment_manifest(p)
    run_manifest_variant(replay, 0, runs_root=tmp_path)
    p['optimizer'] = {'step_rule': 'l-bfgs-b'}
    with pytest.raises(ValueError, match='omit optimizer'):
        parse_experiment_manifest(p)


def test_prepared_generali_rows_are_explicit_and_preserved():
    from experiments.settings import prepare_setting
    from data.loader import eligible_csv_row_indices
    rows = eligible_csv_row_indices('linear')[:4]
    prepared = prepare_setting(dict(source='generali_support', model='glm', row_indices=rows.tolist(),
        quantity='profit', action_grid=[-.1, 0., .1], n_neighbors=2, action_bandwidth=.01))
    assert prepared.truth is None
    np.testing.assert_array_equal(prepared.arrays['row_indices'], rows)
    assert prepared.arrays['predictor_grid'].shape == (4, 3)
    assert prepared.provenance['quantity'] == 'profit'


@pytest.mark.parametrize("keep_marker", [False, True])
def test_resume_missing_endpoint_export_uses_saved_optimizer_theta(tmp_path, monkeypatch, keep_marker):
    manifest = parse_experiment_manifest(payload(tmp_path, 'optimize_with_band'))
    run_manifest_variant(manifest, 0, runs_root=tmp_path)
    folder = manifest.variant_dir(manifest.variants[0], tmp_path)
    if not keep_marker:
        (folder/'band-policy-seed-7.json').unlink()
    endpoint = folder/'seeds/seed-7/band-policy-first_order.npz'
    with np.load(endpoint) as arrays:
        previous = dict(arrays)
    endpoint.unlink()
    monkeypatch.setattr('experiments.execution.execute_experiment_run', lambda *a, **k: pytest.fail('must replay saved theta'))
    assert run_manifest_variant(manifest, 0, runs_root=tmp_path)['n_runs'] == 0
    with np.load(endpoint) as arrays:
        for key in previous:
            np.testing.assert_array_equal(arrays[key], previous[key])
    assert variant_complete(manifest, manifest.variants[0], runs_root=tmp_path)


@pytest.mark.parametrize("keep_marker", [False, True])
def test_changed_policy_endpoints_reject_resume(tmp_path, monkeypatch, keep_marker):
    manifest = parse_experiment_manifest(payload(tmp_path, "optimize_with_band"))
    run_manifest_variant(manifest, 0, runs_root=tmp_path)
    folder = manifest.variant_dir(manifest.variants[0], tmp_path)
    if not keep_marker:
        (folder / "band-policy-seed-7.json").unlink()
    endpoint = folder / "seeds/seed-7/band-policy-first_order.npz"
    with np.load(endpoint) as saved:
        arrays = dict(saved)
    arrays["upper"] += 1
    np.savez_compressed(endpoint, **arrays)
    changed = endpoint.read_bytes()
    monkeypatch.setattr(
        "experiments.execution.execute_experiment_run",
        lambda *args, **kwargs: pytest.fail("must use saved theta"),
    )
    with pytest.raises(ValueError, match="differs|endpoints changed"):
        run_manifest_variant(manifest, 0, runs_root=tmp_path)
    assert endpoint.read_bytes() == changed


def test_partial_seed_resume_rejects_changed_band_evaluation(tmp_path, monkeypatch):
    from experiments import execution

    p = payload(tmp_path, "optimize_with_band")
    p["seeds"]["run_seeds"] = [7, 8]
    manifest = parse_experiment_manifest(p)
    execute = execution.execute_experiment_run
    calls = []

    def interrupt_second_seed(*args, **kwargs):
        calls.append(kwargs["run_context"].run_dir)
        if len(calls) > 1:
            raise RuntimeError("interrupted seed")
        return execute(*args, **kwargs)

    monkeypatch.setattr(execution, "execute_experiment_run", interrupt_second_seed)
    with pytest.raises(RuntimeError, match="interrupted seed"):
        run_manifest_variant(manifest, 0, runs_root=tmp_path)
    folder = manifest.variant_dir(manifest.variants[0], tmp_path)
    assert (folder / "summary-seed-7.json").exists()
    assert not (folder / "summary-seed-8.json").exists()
    recorded = (folder / "band-summary.json").read_bytes()
    evaluation = folder / "band-evaluation.npz"
    with np.load(evaluation) as saved:
        arrays = dict(saved)
    arrays["lower"] -= 1
    np.savez_compressed(evaluation, **arrays)
    with pytest.raises(ValueError, match="differs from recorded provenance"):
        run_manifest_variant(manifest, 0, runs_root=tmp_path)
    assert (folder / "band-summary.json").read_bytes() == recorded
    assert len(calls) == 2


@pytest.mark.parametrize("missing", ["band-summary.json", "band-evaluation.npz"])
def test_missing_band_export_recovers_without_refitting(tmp_path, monkeypatch, missing):
    manifest = parse_experiment_manifest(payload(tmp_path))
    variant = manifest.variants[0]
    run_manifest_variant(manifest, 0, runs_root=tmp_path)
    folder = manifest.variant_dir(variant, tmp_path)
    fitted = (folder / "band.json").read_bytes()
    (folder / missing).unlink()
    monkeypatch.setattr(
        "experiments.band_workflow.fit_band",
        lambda *args, **kwargs: pytest.fail("must load saved fit"),
    )
    assert not variant_complete(manifest, variant, runs_root=tmp_path)
    run_manifest_variant(manifest, 0, runs_root=tmp_path)
    assert variant_complete(manifest, variant, runs_root=tmp_path)
    assert (folder / "band.json").read_bytes() == fitted


def test_changed_observations_invalidate_resume(tmp_path):
    p = payload(tmp_path)
    manifest = parse_experiment_manifest(p)
    run_manifest_variant(manifest, 0, runs_root=tmp_path)
    observations = Path(p["setting"]["path"])
    with np.load(observations) as saved:
        arrays = dict(saved)
    arrays["y"] += 1
    np.savez(observations, **arrays)
    with pytest.raises(ValueError, match="contract changed"):
        run_manifest_variant(manifest, 0, runs_root=tmp_path)


def test_changed_referenced_band_arrays_invalidate_resume(tmp_path):
    p = payload(tmp_path)
    manifest = parse_experiment_manifest(p)
    run_manifest_variant(manifest, 0, runs_root=tmp_path)
    source = manifest.variant_dir(manifest.variants[0], tmp_path) / "band.json"
    p.pop("setting")
    p.update(name="replay", band={"artifact": str(source)})
    replay = parse_experiment_manifest(p)
    run_manifest_variant(replay, 0, runs_root=tmp_path)
    arrays_path = source.parent / json.loads(source.read_text())["arrays"]
    with arrays_path.open("ab") as handle:
        handle.write(b"changed input")
    with pytest.raises(ValueError, match="contract changed"):
        run_manifest_variant(replay, 0, runs_root=tmp_path)


@pytest.mark.parametrize("overrides, message", [
    ({"optimizer": {}}, "omit optimizer"),
    ({"seeds": {}}, "omit optimizer"),
    ({"objective": {"preset": "band_policy"}}, "no optimization objective"),
    ({"objective_modifications": [{"type": "noise"}]}, "no optimization objective"),
    ({"per_seed_plots": True}, "per_seed_plots"),
    ({"truth": {"source": "clean_base_objective"}}, "truth.source=none"),
    ({"band": {}}, "method or a saved artifact"),
    ({"band": {"artifact": "saved.json", "method": "residual_ols"}}, "only band.artifact"),
    ({"band": {"artifact": "saved.json"}}, "omit setting"),
])
def test_band_task_rejects_incompatible_fields(tmp_path, overrides, message):
    p = payload(tmp_path)
    p.update(overrides)
    with pytest.raises(ValueError, match=message):
        parse_experiment_manifest(p)
