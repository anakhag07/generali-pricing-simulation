# Agent Instructions

Project context: pricing simulation and optimization demo. Primary entry point is `main.py`.

## Conventions

- Prefer small, focused changes with clear doc updates.
- Keep the simulation logic deterministic when a seed is set.

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

- Write down lessons from mistakes made to avoid repeating them.
- Keep explicit objective form logic in `src/data/fixed_objective.py` as the single source of truth.

## Validation

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
