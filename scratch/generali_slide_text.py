"""Lean Beamer frames and figure index accompanying the saved-run slide report."""
from pathlib import Path


def _bullets(items):
    return "\\begin{itemize}\n" + "\n".join("\\item " + item for item in items) + "\n\\end{itemize}\n"


def _frame(title, body):
    return "\\begin{frame}{" + title + "}\n\\small\n" + body + "\n\\end{frame}\n\n"


def _plot(title, filename, items):
    return _frame(title, r"\centering" + "\n" +
        r"\includegraphics[width=\linewidth,height=0.64\textheight,keepaspectratio]{figures/" + filename + "}\n" +
        _bullets(items))


def _table(headers, rows):
    return ("\\begin{center}\n\\begin{tabular}{" + "l" + "r"*(len(headers)-1) + "}\n\\toprule\n" +
        " & ".join(headers) + r" \\" + "\n\\midrule\n" +
        "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{center}\n")


def write_slide_text(output, rmse, capture, studies, basis_rmse, basis_metrics, central):
    """Write matching frames with numerical tables and scientific qualifications."""
    percent = lambda name: float(capture.loc[capture.group.eq(name), "capture_percent"].iloc[0])
    frames = _frame("Progress Updates: 11/04", _bullets([
        "Residual analysis and wild-bootstrap bands on Generali insurance data.",
        "Spline acceptance and XGBoost loss predictors throughout the real-data slides.",
        "Saved lower-band policy optimization and sensitivity to residual features.",
        "Synthetic sensitivity to sample size and objective steepness."]))
    frames += _frame("Setup for Wild Bootstrap Band", r"""
\[
 R_i=Y_i-\hat f(x_i,a_i),\qquad
 \tilde g(x,a)=\phi(x,a)^\top\hat\beta.
\]
\[
 w_\delta(x,a)=\hat q_{1-\delta}^{\rm boot}\ell_X(x,a),\quad
 r_{\delta,\rm lower}=[w_\delta-\tilde g]_+,\quad
 r_{\delta,\rm upper}=[w_\delta+\tilde g]_+.
\]
""" + _bullets([
        r"Transferred envelope: $[\hat f-r_{\delta,\rm lower},\;\hat f+r_{\delta,\rm upper}]$.",
        r"Original affine residual basis: $\phi(z,a)=(1,z^\top,a)^\top$, with 11 frozen numeric features.",
        r"1,999 paired Rademacher draws; nominal $1-\delta=95\%$.",
        r"A statement such as $\mathbb P(\mathcal C_\delta\mid\mathcal Z)\geq1-\delta-\rho_n-\tau_n$ requires assumptions; its error terms are not established for these Generali data."]))
    frames += _frame("Setup on Generali Data", r"""
\[
 \mathcal D=\{(x_i,a_i,Y_i)\}_{i=1}^n,\quad
 Y_i=Z_i\bigl((1+a_i)P_i-L_i\bigr),\quad Z_i\in\{0,1\}.
\]
\[
 \hat f(x,a)=\widehat F_{\rm acc}(x,a)
 \bigl((1+a)P(x)-\widehat G_{\rm loss}(x)\bigr).
\]
""" + _bullets([
        r"2024 Portugal auto insurance: 715,023 eligible records; 714,929 distinct customer IDs.",
        r"$\widehat F_{\rm acc}$: monotone spline; $\widehat G_{\rm loss}$: XGBoost.",
        r"Only historical-action outcomes are observed; $f(x,a)=\mathbb E[Y\mid x,a]$ is unknown.",
        r"Original affine band: full-data refit after customer-disjoint 80/20 residual-fit validation.",
        r"Original predictor-training membership is unknown; independent calibration/evaluation and within-customer dependence remain limitations."]))
    frames += _plot("Residual Analysis: Central 95\%", "01_residual_errors_central95.pdf", [
        r"Observed errors $R_i=Y_i-\hat f(x_i,a_i)$ at historical actions.",
        f"Display retains {central['retained_rows']:,} records between the empirical 2.5th and 97.5th percentiles; RMSE below uses untrimmed groups."])
    frames += _plot("Residual Analysis: Affine Fit", "02_affine_residual_fit_holdout.pdf", [
        r"Fit $R_i\approx\phi(z_i,a_i)^\top\beta$ on 572,022 records; validate on 143,001 records.",
        "Points average held-out observed and fitted residuals within 20 equal-count bins; the dashed line is agreement."])
    stage_labels = {"train": "Training", "holdout": "Holdout", "full_refit": "Full-data refit"}
    baseline_table = _table(["Residual-fit stage", "$n$", "Raw RMSE", "After affine fit"], [
        [stage_labels[row.stage], f"{row.n:,}", f"{row.raw_rmse:.4f}", f"{row.fit_rmse:.4f}"]
        for row in rmse.itertuples()])
    (output/"affine_rmse_table.tex").write_text(baseline_table)
    frames += _frame("Residual Analysis: RMSE", baseline_table + _bullets([
        "RMSE compares noisy observed residuals with the zero correction or fitted correction.",
        "It does not measure error against a known true residual-mean function.",
        "The full-data row is in-sample; holdout evaluation precedes the full-data band refit."]))
    frames += _plot("Mean Wild-Bootstrap Band: Policy Range", "03_mean_wild_band_policy_range.pdf", [
        r"Equal-row means over all 715,023 contexts at each common action $a\in[-0.1,0.2]$.",
        "Transfer each customer's endpoints before averaging; this is not a bootstrap band fitted to the population average."])
    frames += _plot("Mean Wild-Bootstrap Band: Wider Range", "04_mean_wild_band_wide_range.pdf", [
        r"The same saved affine band, queried over $a\in[-1,2]$ without recalibration.",
        r"This is an extrapolation diagnostic: spline acceptance is flat below $a=0$ and uses its saved clipped tangent above $a=0.16$."])
    frames += _plot("Wild-Bootstrap Band by Customer Elasticity", "05_customers_by_elasticity.pdf", [
        r"Same three representatives as before: median ranks within low/medium/high $|\partial p_{\rm accept}/\partial a|$ tertiles.",
        r"Ranking uses the saved GLM diagnostic at $a_{\rm ref}=0.087615$; displayed curves use spline/XGBoost only."])
    capture_groups = ["All eligible records", "Logged action in [-0.1, 0.2]", "Low elasticity", "Medium elasticity", "High elasticity", "Three displayed customers"]
    capture_table = _table(["Group", "Captured / total", "Captured"], [
        [name, f"{int(row.captured):,} / {int(row.n):,}", f"{row.capture_percent:.2f}\\%"]
        for name in capture_groups for row in capture[capture.group.eq(name)].itertuples()])
    (output/"observed_capture_table.tex").write_text(capture_table)
    frames += _frame("Observed-Profit Capture", capture_table + _bullets([
        "Each observed green point is tested against its own envelope at its logged action.",
        r"This is in-sample realized-outcome capture, not empirical coverage of the unknown mean $f(x,a)$.",
        "Binary renewal/churn and loss-target variation can place outcomes far from expected profit."]))
    frames += _plot("Optimized Policies", "06_nominal_and_affine_policy_actions.pdf", [
        r"Saved personalized policies on $[-0.1,0.2]$: nominal-profit versus original affine lower-band objective.",
        "Both original spline policy runs reached the 1,000-iteration budget; plotted actions are saved solver endpoints, not certified optima."])
    frames += _frame(r"Dependence on $\phi$: Setup", _bullets([
        r"Matched sensitivity cohort: 715,013 records after the previously fixed ten-record exclusion.",
        r"Customer-disjoint split: 286,015 representation / 286,001 linear-head fit / 142,997 holdout.",
        r"Affine $(1,z,a)$; add $a^2$; then add $a^2$ and $a^4$ (no cubic or interaction terms).",
        r"MLP: 12 standardized inputs, two 16-unit tanh layers, frozen features and a 17-coefficient linear head.",
        r"All four use the same 1,999 paired draws and coefficient-ellipsoid calibration, distinct from the original full-data affine calibration.",
        r"Policies reuse the same 20,000 optimization rows, $[-0.1,0.2]$ action bounds and mean-acceptance floor 0.87877453."]))
    frames += _plot(r"Dependence on $\phi$: Band Width", "07_action_power_mlp_band_widths.pdf", [
        "Mean residual half-width across all 715,013 retained contexts, conditional on each saved feature map.",
        "MLP representation training stopped at 1,000 iterations; these are saved budget-limited features."])
    frames += _plot(r"Dependence on $\phi$: Policies", "08_action_power_mlp_policy_actions.pdf", [
        "Saved nominal policy and the four policies optimized for their respective lower endpoints.",
        "All four lower-band policy solvers reported termination success; global optimality is not certified."])
    frames += _plot(r"Dependence on $\phi$: Policy Values", "09_action_power_mlp_policy_values.pdf", [
        "Each row scores the same nominal policy and that basis's own policy on all retained contexts.",
        "Lower scores use different bands across rows; compare the paired improvement within each row. Predicted profit uses the common frozen predictor."])
    basis_names = {"affine": "Affine", "action_a2": "$+a^2$", "action_a2_a4": "$+a^2+a^4$", "mlp": "MLP"}
    richer_table = _table(["Residual basis", "Holdout RMSE", "Raw RMSE", "$n$"], [
        [basis_names[row.basis], f"{row.fit_rmse:.4f}", f"{row.raw_rmse:.4f}", f"{row.n:,}"]
        for row in basis_rmse.itertuples()])
    (output/"basis_rmse_table.tex").write_text(richer_table)
    frames += _frame(r"Dependence on $\phi$: Residual Fit", richer_table + _bullets([
        "All rows use the identical 142,997-record holdout at historical actions.",
        "A smaller residual RMSE does not establish narrower bands, valid mean-function coverage or higher realized policy profit."]))
    frames += _frame(r"Synthetic Data: Setup", _bullets([
        r"Quadratic objective $f_s(a)=s(5a-5a^2)$; observations $Y_i=f_s(A_i)+\varepsilon_i$.",
        r"Only the off-center setting is shown: $A_i\sim\mathcal N(10,0.45)$ and $\varepsilon_i\sim\mathcal N(0,1)$.",
        r"Quadratic OLS fit $\hat f$ and 1,999 raw-residual wild-bootstrap draws; nominal 95\% band.",
        r"Sample-size illustrations: $n\in\{75,300,2400\}$ at $s=1$; steepness illustrations: $s\in\{0.25,1,4\}$ at $n=300$.",
        "Saved repetition 0 from the 20-repetition sweep. Nested observations and bootstrap signs are paired across conditions.",
        "Lower-bound endpoints come from the repository Stein optimizer; true and fitted references also replay repository solver outputs."]))
    frames += _plot(r"Dependence on $n$ [Synthetic Data]", "10_synthetic_sample_size.pdf", [
        r"Data remain centered at 10; larger $n$ changes uncertainty and the saved lower-bound solution.",
        "Top: objective, fit and band. Bottom: the logging distribution and observed actions."])
    frames += _plot(r"Dependence on $\nabla\hat f$ [Synthetic Data]", "11_synthetic_steepness.pdf", [
        r"Scale $s$ changes the quadratic signal's steepness while noise and data location stay fixed.",
        "Displayed displacement from truth includes both OLS fitting error and the movement induced by the band."])
    (output/"slide_frames.tex").write_text(frames)
    (output/"slideshow.tex").write_text(r"""\documentclass[aspectratio=169]{beamer}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\usefonttheme{professionalfonts}
\setbeamertemplate{navigation symbols}{}
\begin{document}
\input{slide_frames.tex}
\end{document}
""")
    names = sorted(path.name for path in (output/"figures").glob("*.pdf"))
    lines = ["# Spline/XGBoost slide figures", "", "Vector PDFs are in `figures/`; numerical tables and query coordinates are in `data/`.",
        "`slide_frames.tex` contains pasteable Beamer frames with setup details in bullets; `slideshow.tex` supplies a minimal wrapper.", "",
        *[f"- [{name}](figures/{name})" for name in names], "", "## Numerical checks", "",
        f"Observed-profit capture: {percent('All eligible records'):.4f}% across all eligible rows; "
        f"{percent('Logged action in [-0.1, 0.2]'):.4f}% for logged actions in the policy range. "
        "These are descriptive realized-outcome rates, not conditional-mean coverage.", "",
        "Original affine full-data and trimmed coefficient-ellipsoid studies remain separate. All optimized actions replay saved repository solver outputs. "
        "No new fits, resampling or optimization occur. MLP action-grid samples only render exact frozen-network evaluations; all retained contexts enter each mean.", "",
        "The original spline nominal and affine lower-band policies exhausted 1,000 iterations. The four matched lower-band policies report solver success; "
        "global optimality remains unverified. The MLP representation reached its 1,000-step budget. Original predictor-training membership is unknown.", "",
        "## Reproduction", "", "Activate `simulation_env`, then run the existing residual driver from the source worktree:", "",
        "```bash", "MPLCONFIGDIR=/tmp/generali-slide-mpl OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \\",
        "python scratch/plot_generali_profit_residuals.py --slide-report scratch/generali_slideshow.json \\",
        "  --output-dir <fresh-report-directory>", "```", "",
        "Completed reports are preserved. To revise a figure, archive the previous report with checksums, retain `data/` query caches, and regenerate through the same recipe."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
