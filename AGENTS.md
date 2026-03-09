# Agent Instructions

Project context: pricing simulation and optimization demo. Primary entry point is `main.py`.

## Session Start Checklist

At the beginning of every build session, do the following before editing code:

1. Read `AGENTS.md`.
2. Read `README.md`.
3. Inspect the relevant entry points and neighboring files for the area being changed.
4. Check recent tests related to the feature area.
5. If working in a parallel terminal or worktree, assume other branches may have changed the repo recently; re-check the current branch state before editing.
6. If repo structure or behavior appears inconsistent with `README.md` or `AGENTS.md`, update the docs as part of the task.

## Conventions

- Prefer small, focused changes with clear doc updates.
- Keep the simulation logic deterministic when a seed is set.
- Include short comments or specs for different functions. 
- Prefer vectorized or cached computations when they preserve existing logic.

## Source of Truth Hierarchy

When behavior and documentation disagree, use this priority:

1. Explicit objective logic in `src/data/fixed_objective.py`
2. Current implementation in the relevant source module
3. Tests
4. `README.md`
5. `AGENTS.md`

If lower-priority docs are stale, update them in the same task.

## Organization

Before adding code, inspect the surrounding module structure and choose the narrowest sensible location for the change.

Guidelines:
- Extend an existing module when the responsibility clearly matches.
- Create a new file only when it introduces a new reusable concept or prevents an existing file from becoming overloaded.
- Avoid scattering similar logic across multiple files.
- When adding or moving files, record the new organization in `AGENTS.md`.
- Prefer consistency with existing naming and folder conventions over inventing new structure.

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

## Documentation Sync

Documentation maintenance is part of implementation, not a separate follow-up task.

Update `README.md` whenever a change affects any of the following:
- project structure
- setup or execution steps
- configuration options
- outputs, logging, or reporting behavior
- public APIs
- expected experiment workflow

If no README changes are needed, explicitly verify that the existing README is still accurate.

## AGENTS.md Maintenance

Update `AGENTS.md` when:
- files or folders are added, moved, or removed
- responsibilities of modules change
- development workflow changes
- new recurring pitfalls or lessons are discovered
- new public entry points or reporting paths are introduced

Do not leave organizational knowledge only in code diffs; record durable workflow knowledge here.

## Session End Checklist

Before ending a build session:

1. Run the relevant tests.
2. Run the demo if the change affects runtime behavior.
3. Update `README.md` if behavior, structure, or usage changed.
4. Update `AGENTS.md` if organization or workflow knowledge changed.
5. Summarize the change in a way that another agent can continue from the branch without extra chat context.

## Maintenance
- Update `requirements.txt` if necessary after any changes.
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

## Session Discipline

Agent context may be incomplete or stale across terminals, worktrees, or later sessions. Do not rely on prior chat context as the source of truth.

At the start of each implementation session:
- read `AGENTS.md`
- read `README.md`
- inspect the relevant code paths
- confirm the current branch/task scope

At the end of each implementation session:
- run relevant tests
- update `README.md` if usage or behavior changed
- update `AGENTS.md` if structure or workflow knowledge changed
- leave enough written context in commits/PR notes for another session to continue without chat history

## Modes

### Plan Mode
- Do not edit code.
- Propose implementation approach, file targets, and unit-test structure.
- Call out any expected README or AGENTS updates.

### Build Mode
- Make focused code changes.
- Keep docs in sync with implementation.
- Run validation commands after changes.
- Prepare a concise handoff summary suitable for a PR or next session.