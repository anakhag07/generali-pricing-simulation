# Pinned Design-Bench environment

Use the upstream Design-Baselines environment at commit
`785dbcfa58107bfcc426257a1c2e69d7f71c3c27`. The integration does not copy its
requirements: the checked-out upstream repository remains the source of truth.

Create it from a separate checkout:

```bash
git clone https://github.com/brandontrabucco/design-baselines.git
cd design-baselines
git checkout 785dbcfa58107bfcc426257a1c2e69d7f71c3c27
conda env create -f environment.yml
conda run -n design-baselines python -c \
  "import design_bench, design_baselines, tensorflow"
```

The upstream environment is large because `design-bench` and Design-Baselines
declare broad shared dependencies for all tasks and methods. This repository
does not add to that dependency set and invokes only
`design_baselines.gradient_ascent.gradient_ascent`. MuJoCo/ROBEL may still need
host-specific system libraries. Ordinary project tests never import this
environment.
