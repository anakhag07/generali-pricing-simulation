"""Four PDF reports from saved polynomial fits, certificates and repository policies."""

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
    summary = {'config': p, 'metrics': rows, 'contract_sha256': digest,
               'contract': file_record(project/'contract.json'),
               'table': file_record(project/'summary.csv'), 'plot_data': file_record(project/'plot_data.npz'),
               'plots': [file_record(path) for path in paths],
               'reporting_source': file_record(Path(__file__)), 'policies': policy_records,
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
    fig.suptitle('Quartic residual recovery — fixed observation realization')
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
