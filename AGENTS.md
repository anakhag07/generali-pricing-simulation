# Agent Instructions

Project context: pricing simulation and optimization demo. Primary entry point is `main.py`.

## Core Working Rules

- Prefer small, focused changes with clear doc updates.
- Keep simulation logic deterministic when a seed is set.
- Include short comments or specs for functions when helpful.
- Prefer vectorized or cached computations when they preserve existing logic.
- Do not rely on prior chat context as the source of truth; repo context may be stale across terminals, worktrees, or later sessions.

## Logical Rules

- Prefer optimized math operations from online packages instead of implementing from scratch (for gradient descent methods, objective definitions, etc ...)

## Session Workflow

### Start of Session
Before editing code:

1. Read `AGENTS.md`.
2. Read `README.md`.
3. Inspect the relevant entry points and neighboring files.
4. Check recent tests related to the feature area.
5. Confirm the current branch and task scope.
6. If working in a parallel terminal or worktree, assume other branches may have changed the repo recently and re-check branch state.
7. If repo structure or behavior appears inconsistent with `README.md` or `AGENTS.md`, update docs as part of the task.

### End of Session
Before finishing a build session:

1. Run relevant tests.
2. Run the demo if runtime behavior changed.
3. Update `README.md` if behavior, structure, configuration, or usage changed.
4. Update `AGENTS.md` if organization or workflow knowledge changed.
5. Leave a concise summary so another agent can continue from the branch without chat context.

## Modes

### Plan Mode
- Do not edit code.
- Propose implementation approach, file targets, and unit-test structure.
- Ask whether the proposed test structure is appropriate before implementing tests.
- Call out any expected `README.md` or `AGENTS.md` updates.

### Build Mode
- Make focused code changes.
- Keep docs in sync with implementation.
- Run validation commands after changes.
- Prepare a concise handoff summary suitable for a commit, PR, or later session.

## Source of Truth

When behavior and documentation disagree, use this priority:

1. `src/data/fixed_objective.py`
2. Current implementation in the relevant source module
3. Tests
4. `README.md`
5. `AGENTS.md`

If lower-priority docs are stale, update them in the same task.

## Code Organization

Before adding code, inspect the surrounding module structure and choose the narrowest sensible location for the change.

Guidelines:
- Extend an existing module when responsibilities clearly match.
- Create a new file only when it introduces a reusable concept or prevents an existing file from becoming overloaded.
- Avoid scattering similar logic across multiple files.
- Prefer consistency with existing naming and folder conventions.
- When files or folders are added, moved, removed, or repurposed, record the change in `AGENTS.md`.

### Key Components

- **`src/experiments/reporters.py`**
  - `StepReporter`: protocol for per-step metric logging
  - `FileStepLogger`: writes per-step metrics to `steps.csv` in the run directory
  - `ConsoleReporter`: prints to terminal; per-step output controlled by `verbose`
  - `JsonReporter`, `PlotReporter`: end-of-run summary and visualization

- **`src/experiments/config.py`**
  - `ExperimentConfig` dataclass
  - `verbose: bool = False` controls terminal output of per-step metrics

## Testing

- Add or update small, focused unit tests for each change.
- Keep tests deterministic with explicit seeds.
- Avoid plotting, filesystem I/O, or long-running simulations in tests.
- Prefer testing pure functions and small components.
- Keep tests fast.
- Keep tests in the existing flat `tests/` layout for now.

## Documentation and Maintenance

### README.md
Documentation maintenance is part of implementation, not a separate follow-up task.

Update `README.md` whenever a change affects:
- project structure
- setup or execution steps
- configuration options
- outputs, logging, or reporting behavior
- public APIs
- expected experiment workflow

If no README changes are needed, explicitly verify that the existing README is still accurate.

### AGENTS.md
Update `AGENTS.md` when:
- files or folders are added, moved, removed, or repurposed
- module responsibilities change
- development workflow changes
- new recurring pitfalls or lessons are discovered
- new public entry points or reporting paths are introduced

Do not leave durable organizational knowledge only in code diffs.

### Other Maintenance
- Update `requirements.txt` when dependencies change.
- Re-export public APIs in package `__init__.py` files when modules are added or moved.

## Validation

Always activate the environment before running tests or the demo.

Run the demo after changes:

```bash
conda activate simulation_env
python main.py