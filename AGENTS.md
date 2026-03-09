# Agent Instructions

Project context: pricing simulation and optimization demo. Primary entry point is `main.py`.

## Conventions

- Prefer small, focused changes with clear doc updates.
- Keep the simulation logic deterministic when a seed is set.
- Include short comments or specs for different functions. 
- Prefer vectorized or cached computations when they preserve existing logic.

## Organization

Before adding code, understand the organization of the codebase and see if the area where you are planning to add code makes the most sense. Create a new file or folder if this makes organizational sense. Ask clarifying questions if you are unsure where to add code. Take notes on new file organization when edits are made in AGENTS.md.

### Key Components

- **`src/experiments/reporters.py`**: Contains reporting infrastructure including:
  - `StepReporter` protocol for per-step metric logging
  - `FileStepLogger`: Writes per-step metrics to `steps.csv` in the run directory (always active)
  - `ConsoleReporter`: Prints to terminal; per-step output controlled by `verbose` config flag
  - `JsonReporter`, `PlotReporter`: End-of-run summary and visualization

- **`src/experiments/config.py`**: `ExperimentConfig` dataclass with `verbose: bool` field (default `False`) to control terminal output of per-step metrics. 

## Testing Strategy

- Add small, focused unit tests for each change.
- Keep tests deterministic with explicit seeds; avoid plotting, filesystem I/O, or long-running simulations.
- Prefer testing pure functions and small components; keep tests fast.
- Keep tests in the existing flat `tests/` layout for now.

## Plan Mode

- When in plan mode, propose the unit-test structure and ask whether it is appropriate before implementing tests.

## Required Maintenance

- Update `README.md` and `requirements.txt` if necessary after any changes.
- Re-export public APIs in package `__init__.py` files when modules are added or moved.

## Lessons Learned

- Write down lessons from mistakes made to avoid repeating them. Update them here in AGENTS.md
- Keep explicit objective form logic in `src/data/fixed_objective.py` as the single source of truth.

## Validation

Always activate the environment before running tests or the demo.

Run the demo after changes:

```bash
conda activate simulation_env
python main.py
```

Run tests after changes:

```bash
conda activate simulation_env
pytest -q
```
