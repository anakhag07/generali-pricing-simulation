"""Five saved-state reports for interaction and frozen-head nonlinear comparisons."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil

import numpy as np

from experiments.provenance import file_record, read_json, save_arrays_atomic, write_json_atomic
from .core import wilson_interval
from .polynomial import AdditivePolynomial, residual_metrics, integrated_width
from .polynomial_reporting import _objective_regret, _verify
from .polynomial_study import _models
from .structured_residual import (evaluation_rows, restore_structured_band,
                                  numerical_residual_metrics, coverage_grid)


def _label(spec):
    return rf'$\phi_{{6,{spec["interaction_order"]}}}$' if spec['kind'] == 'polynomial' else 'MLP (frozen head)'


def collect_structured(project):
    """Verify complete artifacts and replay metrics/plots without any fitting or optimization."""
    project = Path(project)
    report_source = project/'reporting-source'/Path(__file__).name
    if report_source.exists() and report_source.read_bytes() != Path(__file__).read_bytes():
        raise ValueError('Archive the prior report before changing its reporting source.')
    contract = read_json(project/'contract.json')
    p, digest = contract['contract']['manifest'], contract['sha256']
    for record in [contract['environment'], *contract['source_snapshot'], *contract['contract']['sources']]:
        _verify(record)
    with np.load(contract['environment']['path'], allow_pickle=False) as z:
        snapshot = dict(z)
    repetitions = []
    for index in range(p['repetitions']):
        record = read_json(project/f'repetitions/rep-{index:03d}.json')
        if record['index'] != index or record['contract_sha256'] != digest:
            raise ValueError('Repetition provenance mismatch.')
        _verify(record['arrays'])
        for fit in record['model_diagnostics'].values():
            if 'arrays' in fit:
                _verify(fit['arrays'])
        repetitions.append(record)
    with np.load(repetitions[0]['arrays']['path'], allow_pickle=False) as z:
        state = dict(z)
    reference = read_json(project/'policies/truth.json')
    if reference['contract_sha256'] != digest:
        raise ValueError('Truth reference provenance mismatch.')
    _verify(reference['arrays'])
    with np.load(reference['arrays']['path'], allow_pickle=False) as z:
        true_actions = z['actions']
    X = snapshot['X'][evaluation_rows(p)]
    truth, residual = _models(p)
    mean, variance = p['design']['action_mean'], p['design']['action_variance']
    span = p['reporting']['action_window_standard_deviations']*np.sqrt(variance)
    grid = np.linspace(mean-span, mean+span, p['reporting']['action_points'])
    # Select actual contexts using input coordinates only, before examining fits.
    selected = []
    for target in [(-1., -1., 0.), (0., 0., 0.), (1., 1., 0.)]:
        distance = np.sum((X-target)**2, axis=1)
        distance[selected] = np.inf
        selected.append(int(np.argmin(distance)))
    coordinates = {'action': grid, 'slice_indices': np.asarray(selected),
                   'slice_contexts': X[selected], 'slice_true': residual(X[selected], grid[:, None])}
    rows, policies = [], []
    for spec in p['models']:
        name = spec['name']
        band = restore_structured_band(p, snapshot, state, name)
        metrics = (residual_metrics(band, residual, mean, variance) if isinstance(residual, AdditivePolynomial)
                   else numerical_residual_metrics(band, residual, mean, variance, p['reporting']['integration_tolerance']))
        widths = integrated_width(band, mean, variance, p['reporting']['integration_tolerance'])
        policy = read_json(project/f'policies/{name}.json')
        if policy['contract_sha256'] != digest or policy['degree'] != name:
            raise ValueError('Policy provenance mismatch.')
        for record in [policy['arrays'], *[a['arrays'] for a in policy['attempts']]]:
            _verify(record)
        with np.load(policy['arrays']['path'], allow_pickle=False) as z:
            actions = z['actions']
        fit = repetitions[0]['model_diagnostics'][name]
        row = {'model': name, 'feature_dimension': fit['head_dimension'], **metrics, **widths,
               **_objective_regret(truth, X, true_actions, actions),
               'training_rows': fit['training_rows'], 'calibration_rows': fit['calibration_rows'],
               'evaluation_contexts': len(X), 'training_rmse': fit['training_rmse'],
               'calibration_rmse': fit.get('calibration_rmse'), 'point_fit_success': fit['success'],
               'point_fit_success_repetitions': sum(r['model_diagnostics'][name]['success'] for r in repetitions),
               'condition_number': fit['condition_number'], 'critical': band.critical,
               'solver_success_contexts': policy['selected_success_count'],
               'global_gap_certified_contexts': policy['global_gap_certified_count'],
               'global_gap_tolerance': policy['global_gap_tolerance'],
               'coverage_repetitions': p['repetitions'], 'coverage_domain': p['target']['actions'],
               'actions_outside_display_window': int(np.sum((actions < grid[0]) | (actions > grid[-1])))}
        for event in ['coefficient', 'auxiliary', 'transferred']:
            values = [r['events'][name][event] for r in repetitions]
            if any(v is None for v in values):
                count, probability, lo, hi = None, None, None, None
            else:
                count = sum(values)
                probability = count/p['repetitions']
                lo, hi = wilson_interval(count, p['repetitions'])
            row.update({event+'_count': count, event+'_coverage': probability,
                        event+'_wilson_lower': lo, event+'_wilson_upper': hi})
        if p['target']['actions'] == 'finite_grid':
            row.update(coverage_action_points=p['target']['grid_points'],
                       coverage_action_min=float(coverage_grid(p)[0]), coverage_action_max=float(coverage_grid(p)[-1]),
                       refinement_action_points=p['target']['refinement_points'],
                       refined_transferred_count=sum(r['events'][name]['refined_transferred'] for r in repetitions))
        rows.append(row)
        policies.append(policy)
        _, _, lower, upper = band.evaluate(grid[:, None])
        slices = restore_structured_band(p, snapshot, state, name, contexts=X[selected])
        fitted, width, _, _ = slices.evaluate(grid[:, None])
        coordinates.update({name+'_slice_fit': fitted, name+'_slice_width': width,
                            name+'_mean_full_width': (lower+upper).mean(1), name+'_actions': actions})
        print(f"{name}: R2={row['r_squared']:.5f}, coverage={row['transferred_count']}/{p['repetitions']}, "
              f"regret={row['regret']:.6f}, fit success={row['point_fit_success_repetitions']}/{p['repetitions']}", flush=True)
    with (project/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_arrays_atomic(project/'plot_data.npz', **coordinates)
    paths = _render(p, rows, coordinates, project, snapshot, state, X, truth, residual)
    report_source.parent.mkdir(exist_ok=True)
    shutil.copyfile(__file__, report_source)
    summary = {'config': p, 'contract_sha256': digest, 'contract': file_record(project/'contract.json'),
               'metrics': rows, 'policies': policies, 'truth_reference': reference,
               'plots': [file_record(path) for path in paths], 'table': file_record(project/'summary.csv'),
               'plot_data': file_record(project/'plot_data.npz'), 'reporting_source': file_record(Path(__file__)),
               'reporting_source_snapshot': file_record(report_source),
               'coverage_domain': p['target'], 'fit_protocol': p['fit_protocol'],
               'coverage_claim': ('Empirical conditional coverage; finite-grid checks do not certify continuum/tails.'
                                  if p['target']['actions'] == 'finite_grid' else
                                  'Empirical conditional coverage, with exact polynomial checks over all real actions.'),
               'metrics_domain': 'Uniform evaluation contexts and independent full-Gaussian actions, no truncation.',
               'calibration_semantics': ('Frozen-training-head protocol never replaces the training head; calibration rows are not independent test data.'
                                         if p['fit_protocol'] == 'frozen_training_head' else
                                         'OLS and raw-residual bootstrap use all 300 observations.'),
               'regret_definition': 'Direct empirical true J at saved repository truth actions minus J at repository Stein endpoints.',
               'observations': file_record(project/'observation_data.csv')}
    write_json_atomic(project/'summary.json', summary)
    return summary


def _render(p, rows, data, project, snapshot, state, X, truth, residual):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.titlesize': 16,
                         'font.weight': 'normal', 'axes.titleweight': 'normal'})
    paths, names = [], [_label(m) for m in p['models']]
    def save(fig, name):
        path = project/name
        fig.savefig(path, format='pdf')
        plt.close(fig)
        paths.append(path)
    fig, axes = plt.subplots(3, len(rows), figsize=(5*len(rows), 11), sharex=True, squeeze=False, constrained_layout=True)
    fig.suptitle('True residual recovery at fixed calibration contexts\n'
                 'Scores: all evaluation contexts and the full Gaussian action distribution')
    for j, row in enumerate(rows):
        name = row['model']
        for i in range(3):
            ax = axes[i, j]
            fit = data[name+'_slice_fit'][:, i]
            ax.plot(data['action'], data['slice_true'][:, i], label='True residual')
            ax.plot(data['action'], fit, label='Fitted residual')
            context = data['slice_contexts'][i]
            title = names[j]+f'\nx1={context[0]:.2f}, x2={context[1]:.2f}'
            if i == 0:
                title += f'\nOverall R²={row["r_squared"]:.3f}, RMSE={row["rmse"]:.3f}'
                if not row['point_fit_success']:
                    title += '\nTraining solver did not converge'
            ax.set_title(title)
            ax.set_xlabel('Action a')
            ax.set_ylabel('Residual at selected context')
            ax.legend()
    save(fig, '01_residual_recovery.pdf')
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    coverage = np.array([r['transferred_coverage'] for r in rows])
    lo, hi = np.array([r['transferred_wilson_lower'] for r in rows]), np.array([r['transferred_wilson_upper'] for r in rows])
    ax.bar(positions, coverage)
    ax.errorbar(positions, coverage, yerr=np.stack((coverage-lo, hi-coverage)), fmt='none', color='black', capsize=4, label='95% Wilson interval')
    ax.axhline(1-p['bootstrap']['delta'], linestyle='--', label='Nominal 95%')
    labels = [name+f'\n{row["transferred_count"]}/{p["repetitions"]}' for name, row in zip(names, rows)]
    if p['target']['actions'] == 'finite_grid':
        labels = [label+f'\nRefined: {r["refined_transferred_count"]}/{p["repetitions"]}' for label, r in zip(labels, rows)]
        grid = coverage_grid(p)
        ax.set_title(f'Approximate grid coverage: {len(X)} contexts × {len(grid)} action points\n'
                     f'a in [{grid[0]:.2f}, {grid[-1]:.2f}]; refinement: {p["target"]["refinement_points"]} points')
    else:
        ax.set_title(f'Asymmetric envelope: {len(X)} contexts, all real actions')
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel('Residual representation (covered repetitions / total)')
    ax.set_ylabel('Simultaneous coverage probability')
    ax.legend()
    save(fig, '02_simultaneous_coverage.pdf')
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name, row in zip(names, rows):
        ax.plot(data['action'], data[row['model']+'_mean_full_width'], label=name+f': weighted mean={row["mean_full_width"]:.3f}')
    ax.set_xlabel('Action a')
    ax.set_ylabel('Mean full transferred-envelope width')
    ax.set_title('Asymmetric envelope width — fixed observation realization')
    ax.legend()
    save(fig, '03_average_envelope_width.pdf')
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.bar(positions, [r['regret'] for r in rows])
    ax.bar_label(bars, labels=[f'{r["regret"]:.5f}' for r in rows], padding=3, fontsize=10)
    ax.set_ylim(0, 1.2*max(r['regret'] for r in rows) or .01)
    ax.set_xticks(positions, [name+f'\nRMS={r["rms_action_error"]:.3f}' for name, r in zip(names, rows)])
    ax.set_xlabel('Residual representation (RMS action error from 0.5)')
    ax.set_ylabel('Mean true-function regret')
    ax.set_title(r'Regret: $J(\theta^\star)-J(\widehat{\theta})$, $\theta^\star=0.5$')
    save(fig, '04_policy_regret.pdf')
    reference = X.mean(0)
    aligned = state['observations']-(snapshot['X']-reference)@np.asarray(truth.context)
    with (project/'observation_data.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['action', 'aligned_observation', 'noise_sd'])
        writer.writerows(zip(snapshot['actions'], aligned, snapshot['sigma']))
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.suptitle('True function, fixed surrogate and observation noise')
    scatter = ax.scatter(snapshot['actions'], aligned, c=snapshot['sigma'], cmap='viridis',
                         vmin=p['noise']['sigma_min'], vmax=p['noise']['sigma_max'], label='Saved observations (context aligned)')
    grid = np.linspace(min(data['action'][0], snapshot['actions'].min()), max(data['action'][-1], snapshot['actions'].max()), len(data['action']))
    ax.plot(grid, truth(reference, grid), label=r'True $f(\bar{x},a)$', zorder=3)
    ax.plot(grid, truth(reference, grid)-residual(reference, grid), label=r'Surrogate $\widehat f(\bar{x},a)$', zorder=3)
    ax.set_xlabel('Action a')
    ax.set_ylabel('Response aligned to mean context')
    ax.set_title('Original observations with additive context effects aligned')
    ax.legend()
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label(r'Original noise SD $\sigma(x_i,a_i)$')
    save(fig, '05_truth_surrogate_and_observations.pdf')
    return paths
