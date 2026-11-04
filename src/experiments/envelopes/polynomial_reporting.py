"""Comparison and observation PDFs from saved polynomial fits, data and policies."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.provenance import file_record, read_json, save_arrays_atomic, write_json_atomic
from .core import wilson_interval
from .polynomial import residual_metrics, integrated_width
from .polynomial_study import restore_band, _models


def collect_polynomial(project):
    """Verify saved artifacts and report them without fitting, sampling or optimization."""
    project = Path(project)
    contract = read_json(project/'contract.json')
    digest = contract['sha256']
    p = contract['contract']['manifest']
    _verify(contract['environment'])
    for source in contract['source_snapshot']:
        _verify(source)
    # Queries must still use the original verified mathematical implementation.
    math_source = next(s for s in contract['contract']['sources'] if s['path'].endswith('/envelopes/polynomial.py'))
    if math_source['sha256'] != file_record(Path(__file__).with_name('polynomial.py'))['sha256']:
        raise ValueError('Numerical query implementation differs from the fitted-state contract.')
    with np.load(project/'inputs.npz', allow_pickle=False) as saved:
        snapshot = dict(saved)
    repetitions = []
    for index in range(p['repetitions']):
        path = project/'repetitions'/f'rep-{index:03d}.json'
        record = read_json(path)
        if record['contract_sha256'] != digest or record['index'] != index:
            raise ValueError('Incompatible repetition contract/index.')
        _verify(record['arrays'])
        repetitions.append(record)
    with np.load(project/'repetitions/rep-000.npz', allow_pickle=False) as saved:
        state = dict(saved)
    reference = read_json(project/'policies/truth.json')
    if reference['contract_sha256'] != digest:
        raise ValueError('Incompatible truth reference.')
    _verify(reference['arrays'])
    with np.load(reference['arrays']['path'], allow_pickle=False) as saved:
        true_actions = saved['actions']
    truth, residual = _models(p)
    mean, variance = p['design']['action_mean'], p['design']['action_variance']
    span = p['reporting']['action_window_standard_deviations']*np.sqrt(variance)
    grid = np.linspace(mean-span, mean+span, p['reporting']['action_points'])
    coordinates = {'action': grid, 'mean_true_residual': residual(snapshot['X'], grid[:, None]).mean(axis=1)}
    rows, policy_records = [], []
    for degree in p['feature_degrees']:
        band = restore_band(snapshot, state, degree)
        metrics = residual_metrics(band, residual, mean, variance)
        widths = integrated_width(band, mean, variance, p['reporting']['integration_tolerance'])
        policy = read_json(project/'policies'/f'degree-{degree}.json')
        if policy['contract_sha256'] != digest:
            raise ValueError('Incompatible policy contract.')
        _verify(policy['arrays'])
        for attempt in policy['attempts']:
            _verify(attempt['arrays'])
        with np.load(policy['arrays']['path'], allow_pickle=False) as saved:
            actions = saved['actions']
        objective_metrics = _objective_regret(truth, snapshot['X'], true_actions, actions)
        regret = objective_metrics['regret']
        row = {'degree': degree, 'feature_dimension': 4+degree, 'headline_repetition': 0,
               **metrics, **widths, **objective_metrics,
               'mean_action': float(actions.mean()), 'min_action': float(actions.min()), 'max_action': float(actions.max()),
               'max_leverage': float(state[f'degree{degree}_leverage'].max()),
               'design_condition_number': float(state[f'degree{degree}_condition_number']),
               'critical': float(state[f'degree{degree}_critical']),
               'solver_success_contexts': policy['selected_success_count'],
               'global_gap_certified_contexts': policy['global_gap_certified_count'],
               'global_gap_tolerance': policy['global_gap_tolerance'], 'contexts': p['n'],
               'coverage_repetitions': p['repetitions']}
        for event in ('coefficient', 'auxiliary', 'transferred'):
            count = sum(int(rep['events'][str(degree)][event]) for rep in repetitions)
            lo, hi = wilson_interval(count, p['repetitions'])
            row.update({f'{event}_count': count, f'{event}_coverage': count/p['repetitions'],
                        f'{event}_wilson_lower': lo, f'{event}_wilson_upper': hi})
        rows.append(row)
        policy_records.append(policy)
        fitted, width, lower, upper = band.evaluate(grid[:, None])
        coordinates.update({f'degree{degree}_mean_fit': fitted.mean(axis=1),
                            f'degree{degree}_mean_half_width': width.mean(axis=1),
                            f'degree{degree}_mean_lower_radius': lower.mean(axis=1),
                            f'degree{degree}_mean_upper_radius': upper.mean(axis=1),
                            f'degree{degree}_mean_full_width': (lower+upper).mean(axis=1),
                            f'degree{degree}_actions': actions})
        print(f"Degree {degree}: R2={metrics['r_squared']:.5f}, RMSE={metrics['rmse']:.5f}, "
              f"coverage={row['transferred_count']}/{p['repetitions']}, regret={regret:.6f}", flush=True)
    with (project/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_arrays_atomic(project/'plot_data.npz', **coordinates)
    paths = render_polynomial_reports(p, rows, coordinates, project)
    observations = write_observation_report(project)
    paths.append(Path(observations['pdf']['path']))
    summary = {'config': p, 'metrics': rows, 'contract_sha256': digest,
               'contract': file_record(project/'contract.json'),
               'table': file_record(project/'summary.csv'), 'plot_data': file_record(project/'plot_data.npz'),
               'plots': [file_record(path) for path in paths],
               'reporting_source': file_record(Path(__file__)), 'policies': policy_records,
               'observation_report': file_record(project/'observation_report.json'),
               'truth_reference': reference,
               'coverage_event': 'f inside asymmetric predictor envelope at all 300 saved contexts and every real action',
               'coverage_claim': 'Empirical conditional coverage; nominal 95% is not a finite-sample guarantee.',
               'metrics_domain': 'All saved contexts uniformly; independent A~N(0.5,0.45), no truncation.',
               'width_definition': 'Full transferred width: mean(lower radius + upper radius), after per-context transfer.',
               'regret_definition': 'J(theta_star)-J(theta_hat), where J is the empirical mean true objective over the saved contexts.',
               'numerics': 'Exact rational sign certificates for saved floating coefficients; not enclosures of fitting roundoff.'}
    write_json_atomic(project/'summary.json', summary)
    return summary


def render_polynomial_reports(p, rows, data, project):
    """Render exactly four default-style vector PDFs from verified saved coordinates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.titlesize': 16,
                         'font.weight': 'normal', 'axes.titleweight': 'normal'})
    project = Path(project)
    paths = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True, sharey=True, constrained_layout=True)
    residual_degree = len(p['residual']['action'])-1
    label = {1: 'Linear', 2: 'Quadratic', 4: 'Quartic', 6: 'Sextic'}.get(residual_degree, f'Degree-{residual_degree}')
    fig.suptitle(f'{label} residual recovery — fixed observation realization')
    for ax, row in zip(axes.flat, rows):
        k = row['degree']
        fit, width = data[f'degree{k}_mean_fit'], data[f'degree{k}_mean_half_width']
        ax.plot(data['action'], data['mean_true_residual'], label='True residual')
        line, = ax.plot(data['action'], fit, label='Fitted residual')
        ax.fill_between(data['action'], fit-width, fit+width, color=line.get_color(), alpha=.2,
                        label='Mean residual-band endpoints')
        ax.set_title(rf'$\phi_{k}$: $R^2={row["r_squared"]:.3f}$, RMSE = {row["rmse"]:.3f}')
        ax.set_xlabel('Action a')
        ax.set_ylabel('Mean residual across contexts')
        ax.legend()
    paths.append(project/'01_residual_recovery.pdf')
    fig.savefig(paths[-1], format='pdf')
    plt.close(fig)

    degrees = np.array([row['degree'] for row in rows])
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    coverage = np.array([row['transferred_coverage'] for row in rows])
    lower = np.array([row['transferred_wilson_lower'] for row in rows])
    upper = np.array([row['transferred_wilson_upper'] for row in rows])
    ax.bar(degrees, coverage)
    ax.errorbar(degrees, coverage, yerr=np.stack((coverage-lower, upper-coverage)),
                fmt='none', color='black', capsize=4, label='95% Wilson interval')
    ax.axhline(1-p['bootstrap']['delta'], linestyle='--', label='Nominal 95%')
    ax.set_xticks(degrees, [rf'$\phi_{r["degree"]}$'+f'\n{r["transferred_count"]}/{p["repetitions"]}' for r in rows])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel('Residual feature basis (covered repetitions / total)')
    ax.set_ylabel('Simultaneous coverage probability')
    ax.set_title('Asymmetric envelope: all saved contexts, all real actions')
    ax.legend()
    paths.append(project/'02_simultaneous_coverage.pdf')
    fig.savefig(paths[-1], format='pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for row in rows:
        k = row['degree']
        ax.plot(data['action'], data[f'degree{k}_mean_full_width'],
                label=rf'$\phi_{k}$: weighted mean = {row["mean_full_width"]:.3f}')
    ax.set_xlabel('Action a')
    ax.set_ylabel('Mean full transferred-envelope width')
    ax.set_title('Asymmetric envelope width — fixed observation realization')
    ax.legend()
    paths.append(project/'03_average_envelope_width.pdf')
    fig.savefig(paths[-1], format='pdf')
    plt.close(fig)

    paths.append(_render_policy_regret(rows, project))
    return paths


def _render_policy_regret(rows, project):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.titlesize': 16,
                         'font.weight': 'normal', 'axes.titleweight': 'normal'})
    degrees = np.array([row['degree'] for row in rows])
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.bar(degrees, [row['regret'] for row in rows])
    ax.bar_label(bars, labels=[f'{row["regret"]:.5f}' for row in rows], padding=3, fontsize=10)
    ax.set_ylim(0, 1.2*max(row['regret'] for row in rows) or .01)
    ax.set_xticks(degrees, [rf'$\phi_{r["degree"]}$'+f'\nRMS = {r["rms_action_error"]:.3f}' for r in rows])
    ax.set_xlabel('Residual feature basis (RMS action error from 0.5)')
    ax.set_ylabel('Mean true-function regret')
    ax.set_title(r'Regret: $J(\theta^\star)-J(\widehat{\theta})$, $\theta^\star=0.5$')
    path = Path(project)/'04_policy_regret.pdf'
    fig.savefig(path, format='pdf')
    plt.close(fig)
    return path


def _objective_regret(truth, contexts, true_actions, actions):
    optimal = float(np.mean(truth(contexts, true_actions)))
    converged = float(np.mean(truth(contexts, actions)))
    regret = optimal-converged
    rms = float(np.sqrt(np.mean((actions-.5)**2)))
    if not np.isclose(regret, .5*rms*rms, atol=1e-12, rtol=1e-10):
        raise ValueError('Direct objective regret disagrees with the quadratic identity.')
    return {'objective_at_theta_star': optimal, 'objective_at_theta_hat': converged,
            'regret': regret, 'rms_action_error': rms}


def update_regret_report(project):
    """Replay only direct true-objective regret, its PDF and table from saved policies."""
    project = Path(project)
    summary = read_json(project/'summary.json')
    _verify(summary['table'])
    _verify(summary['contract'])
    for plot in summary['plots']:
        _verify(plot)
    contract = read_json(project/'contract.json')
    if contract['sha256'] != summary['contract_sha256'] or contract['contract']['manifest'] != summary['config']:
        raise ValueError('Report and numerical contract differ.')
    _verify(contract['environment'])
    _verify(summary['truth_reference']['arrays'])
    with np.load(contract['environment']['path'], allow_pickle=False) as saved:
        contexts = saved['X']
    with np.load(summary['truth_reference']['arrays']['path'], allow_pickle=False) as saved:
        true_actions = saved['actions']
    truth, _ = _models(summary['config'])
    for row, policy in zip(summary['metrics'], summary['policies']):
        if row['degree'] != policy['degree'] or policy['contract_sha256'] != summary['contract_sha256']:
            raise ValueError('Policy and summary provenance differ.')
        _verify(policy['arrays'])
        with np.load(policy['arrays']['path'], allow_pickle=False) as saved:
            row.update(_objective_regret(truth, contexts, true_actions, saved['actions']))
    with (project/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(summary['metrics'][0]))
        writer.writeheader()
        writer.writerows(summary['metrics'])
    path = _render_policy_regret(summary['metrics'], project)
    summary['plots'] = [file_record(path) if Path(record['path']) == path else record for record in summary['plots']]
    summary['table'] = file_record(project/'summary.csv')
    summary['regret_definition'] = 'J(theta_star)-J(theta_hat), where J is the empirical mean true objective over the saved contexts.'
    summary['regret_reporting_source'] = file_record(Path(__file__))
    write_json_atomic(project/'summary.json', summary)
    return summary


def _verify(record):
    if file_record(record['path']) != record:
        raise ValueError(f"Saved artifact checksum differs: {record['path']}")


def write_metric_comparison(studies, output_dir):
    """Replay saved R2, mean transferred radius and regret into one three-panel PDF per study.

    Each study specifies project, name, title, and optionally models and selection_note.
    """
    import shutil
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir).resolve()
    source = Path(__file__).resolve()
    snapshot = output_dir/'reporting-source'/source.name
    if snapshot.exists() and snapshot.read_bytes() != source.read_bytes():
        raise ValueError('Archive the previous metric comparison before changing its source.')
    request = {'studies': studies, 'output_dir': str(output_dir)}
    request_path = output_dir/'request.json'
    if request_path.exists() and read_json(request_path) != request:
        raise ValueError('Archive the previous metric comparison before changing its selection.')

    points, inputs, groups = [], [], []
    for study in studies:
        project = Path(study['project']).resolve()
        summary = read_json(project/'summary.json')
        records = [file_record(project/'summary.json'), summary['contract'], summary['table']]
        contract = read_json(summary['contract']['path'])
        if contract['sha256'] != summary['contract_sha256']:
            raise ValueError('Summary and fitted-state contracts differ.')
        records.append(contract['environment'])
        records.append(summary['truth_reference']['arrays'])
        configs = {spec['name']: spec for spec in summary['config'].get('models', [])}
        selected = study.get('models')
        group = []
        for row, policy in zip(summary['metrics'], summary['policies'], strict=True):
            model = row.get('model', f"degree-{row.get('degree')}")
            if selected is not None and model not in selected:
                continue
            policy_path = project/'policies'/f'{model}.json'
            if read_json(policy_path) != policy or policy['contract_sha256'] != summary['contract_sha256']:
                raise ValueError(f'Policy provenance differs for {model}.')
            records.extend([file_record(policy_path), policy['arrays']])
            records.extend(attempt['arrays'] for attempt in policy['attempts'])
            spec = configs.get(model)
            if spec is None:
                label = rf'$\phi_{{{row["degree"]}}}$'
            elif spec['kind'] == 'polynomial':
                label = rf'$\phi_{{{spec["degree"]},{spec["interaction_order"]}}}$'
            else:
                label = f'MLP {spec["hidden_width"]}×{spec["hidden_width"]} ({spec["max_steps"]:,}-step limit)'
            contexts = row.get('evaluation_contexts', row.get('contexts'))
            if row['solver_success_contexts'] < contexts:
                label += ' (policy not fully converged)'
            radius = (row['mean_lower_radius']+row['mean_upper_radius'])/2
            if not np.isclose(2*radius, row['mean_full_width'], rtol=1e-12, atol=1e-12):
                raise ValueError('Asymmetric radii do not match the saved full width.')
            point = {'study': study['name'], 'model': model, 'label': label,
                     'headline_repetition': row.get('headline_repetition', 0),
                     'r_squared': row['r_squared'], 'mean_radius': radius,
                     'mean_lower_radius': row['mean_lower_radius'],
                     'mean_upper_radius': row['mean_upper_radius'],
                     'mean_full_width': row['mean_full_width'], 'regret': row['regret'],
                     'evaluation_contexts': contexts,
                     'solver_success_contexts': row['solver_success_contexts'],
                     'source_project': str(project)}
            group.append(point)
        if not group or (selected is not None and set(selected) != {p['model'] for p in group}):
            raise ValueError('Selected models are missing from the saved report.')
        for record in records:
            _verify(record)
        inputs.extend(records)
        points.extend(group)
        groups.append((study, group))

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    style = {'figure.titlesize': 16, 'axes.titlesize': 14, 'axes.labelsize': 12,
             'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10}
    radius_label = r'Mean transferred radius $E[(r_l+r_u)/2]$'
    r2_label = r'True-residual $R^2$ (full Gaussian actions)'
    regret_label = r'True regret $J(\theta^\star)-J(\widehat\theta)$'
    panels = [('mean_radius', 'r_squared', radius_label, r2_label, r'$R^2$ vs. mean radius'),
              ('mean_radius', 'regret', radius_label, regret_label, 'Regret vs. mean radius'),
              ('r_squared', 'regret', r2_label, regret_label, r'Regret vs. $R^2$')]
    with plt.rc_context(style):
        for study, group in groups:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
            fig.suptitle(study['title']+' — fixed noise realization')
            for ax, (x, y, xlabel, ylabel, title) in zip(axes, panels):
                for point in group:
                    ax.plot(point[x], point[y], marker='o', linestyle='none', label=point['label'])
                ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center', ncols=len(group))
            path = output_dir/f'{study["name"]}_r_squared_radius_regret.pdf'
            fig.savefig(path, format='pdf')
            plt.close(fig)
            paths.append(path)

    table = output_dir/'metric_comparison.csv'
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    snapshot.parent.mkdir(exist_ok=True)
    shutil.copyfile(source, snapshot)
    write_json_atomic(request_path, request)
    report = {'request': file_record(request_path), 'inputs': inputs,
              'reporting_source': file_record(source), 'reporting_source_snapshot': file_record(snapshot),
              'plots': [file_record(path) for path in paths], 'table': file_record(table),
              'metrics': points,
              'radius_definition': 'Mean(lower asymmetric radius + upper asymmetric radius)/2; half transferred full width.',
              'r_squared_definition': 'Known noiseless residual, uniform evaluation contexts and full Gaussian actions.',
              'regret_definition': 'Saved direct true-objective difference at repository optimizer endpoints.',
              'comparison_semantics': 'All coordinates use repetition 0; within-study comparisons, not matched sample sizes across studies.'}
    write_json_atomic(output_dir/'provenance.json', report)
    return report


def write_observation_report(project):
    """Replay true/surrogate curves and context-aligned observations with saved noise scales."""
    from scipy.integrate import quad
    from scipy.stats import norm
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    project = Path(project).resolve()
    contract = read_json(project/'contract.json')
    repetition = read_json(project/'repetitions/rep-000.json')
    if repetition['contract_sha256'] != contract['sha256']:
        raise ValueError('Observation and environment contracts differ.')
    _verify(contract['environment'])
    _verify(repetition['arrays'])
    original_math = next(s for s in contract['contract']['sources']
                         if s['path'].endswith('/envelopes/polynomial.py'))
    if original_math['sha256'] != file_record(Path(__file__).with_name('polynomial.py'))['sha256']:
        raise ValueError('Mathematical query source differs from the saved contract.')
    with np.load(contract['environment']['path'], allow_pickle=False) as saved:
        snapshot = dict(saved)
    with np.load(repetition['arrays']['path'], allow_pickle=False) as saved:
        state = dict(saved)
    p = contract['contract']['manifest']
    truth, residual = _models(p)
    X, actions, sigma = snapshot['X'], snapshot['actions'], snapshot['sigma']
    reference = X.mean(axis=0)
    offsets = (X-reference) @ np.asarray(truth.context)
    aligned = state['observations']-offsets
    noise = state['observations']-truth(X, actions)
    if not np.allclose(noise, sigma*state['innovations'], atol=2e-14, rtol=1e-13):
        raise ValueError('Saved noise no longer agrees with observation construction.')
    if not np.allclose(aligned, truth(reference, actions)+noise, atol=2e-14, rtol=1e-13):
        raise ValueError('Context alignment does not preserve the original noise.')
    mean, variance = p['design']['action_mean'], p['design']['action_variance']
    sd = np.sqrt(variance)
    span = p['reporting']['action_window_standard_deviations']*sd
    grid = np.linspace(min(mean-span, actions.min()), max(mean+span, actions.max()),
                       p['reporting']['action_points'])
    truth_curve = truth(reference, grid)
    predictor_curve = truth_curve-residual(reference, grid)
    csv_path = project/'observation_data.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row', 'x1', 'x2', 'x3', 'action', 'observation',
                         'context_offset', 'aligned_observation', 'noise', 'noise_sd'])
        for i in range(len(X)):
            writer.writerow([i, *X[i], actions[i], state['observations'][i],
                             offsets[i], aligned[i], noise[i], sigma[i]])
    curves_path = project/'observation_curves.csv'
    with curves_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['action', 'true_function_at_mean_context', 'surrogate_at_mean_context'])
        writer.writerows(zip(grid, truth_curve, predictor_curve))

    plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.titlesize': 16,
                         'font.weight': 'normal', 'axes.titleweight': 'normal'})
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.suptitle('True function, fixed surrogate and observation noise')
    colors = {'c': sigma, 'cmap': 'viridis', 'vmin': p['noise']['sigma_min'],
              'vmax': p['noise']['sigma_max']}
    scatter = ax.scatter(actions, aligned, label='Saved observations (context aligned)', **colors)
    ax.plot(grid, truth_curve, label=r'True $f(\bar{x},a)$', zorder=3)
    ax.plot(grid, predictor_curve, label=r'Surrogate $\widehat{f}(\bar{x},a)$', zorder=3)
    ax.set_xlabel('Action a')
    ax.set_ylabel('Response aligned to mean context')
    ax.set_title('Original observations with additive context effects aligned')
    ax.legend()
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label(r'Original noise SD $\sigma(x_i,a_i)$', fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    pdf = project/'05_truth_surrogate_and_observations.pdf'
    fig.savefig(pdf, format='pdf')
    plt.close(fig)

    band = restore_band(snapshot, state, 1)
    metrics = residual_metrics(band, residual, mean, variance)
    def integrand(z):
        density = norm.pdf(z)
        if density == 0:
            return 0.
        action = mean+sd*z
        return float(np.mean((residual(X, action)-band.evaluate(action)[0])**2)*density)
    window_sd = p['reporting']['action_window_standard_deviations']
    inside, inside_error = quad(integrand, -window_sd, window_sd, epsabs=1e-12)
    left, left_error = quad(integrand, -np.inf, -window_sd, epsabs=1e-12)
    right, right_error = quad(integrand, window_sd, np.inf, epsabs=1e-12)
    if not np.isclose(inside+left+right, metrics['mse'], atol=1e-11, rtol=1e-10):
        raise ValueError('Weighted error decomposition differs from exact Gaussian moments.')
    report = {'contract_sha256': contract['sha256'], 'repetition': 0,
              'input_environment': contract['environment'], 'input_observations': repetition['arrays'],
              'reference_context': reference.tolist(), 'alignment': 'Y_i - c^T(x_i - mean(X)), with c the true additive context coefficients.',
              'noise_interpretation': 'Original sigma(x_i,a_i) and innovations retained; no off-design GP draw or sample recentering.',
              'noise_sample_mean': float(noise.mean()), 'noise_sd_min': float(sigma.min()), 'noise_sd_max': float(sigma.max()),
              'pdf': file_record(pdf), 'observations_csv': file_record(csv_path), 'curves_csv': file_record(curves_path),
              'reporting_source': file_record(Path(__file__)),
              'phi1_r_squared_diagnostics': {**metrics, 'display_window': [mean-span, mean+span],
                  'action_probability_outside': float(2*norm.sf(window_sd)),
                  'mse_inside_contribution': inside, 'mse_outside_contribution': left+right,
                  'mse_fraction_outside': (left+right)/metrics['mse'],
                  'quadrature_error_sum': inside_error+left_error+right_error,
                  'context_component_variance': float(np.var(X @ np.asarray(residual.context)))}}
    write_json_atomic(project/'observation_report.json', report)
    return report
