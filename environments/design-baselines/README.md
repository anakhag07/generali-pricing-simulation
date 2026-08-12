# Pinned Design-Bench environment

This environment mirrors the full upstream Design-Baselines environment at
commit `785dbcfa58107bfcc426257a1c2e69d7f71c3c27`. Its requirements pin
`design-bench[all]==2.0.20`, `morphing-agents==1.5.1`, TensorFlow 2.3.2, and
TensorFlow Probability 0.11.0. The final editable upstream package entry was
replaced by a Git URL pinned to that exact commit.

Create it from the repository root:

```bash
conda env create -f environments/design-baselines/environment.yml
conda run -n design-baselines python -c \
  "import design_bench, design_baselines, tensorflow"
```

Then point the modern bridge at the environment executable:

```bash
export DESIGN_BENCH_PYTHON="$(conda run -n design-baselines which python)"
```

This is intentionally a large legacy environment: upstream installs the
dependencies for every baseline even though this integration invokes only
`design_baselines.gradient_ascent.gradient_ascent`. MuJoCo/ROBEL may also need
host-specific system libraries and a compatible GPU driver. Ordinary project
tests do not create or import this environment.
