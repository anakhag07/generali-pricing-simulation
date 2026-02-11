# Agent Instructions

Project context: pricing simulation and optimization demo. Primary entry point is `main.py`.

## Conventions

- Prefer small, focused changes with clear doc updates.
- Keep the simulation logic deterministic when a seed is set.

## Required Maintenance

- Update `README.md` and `requirements.txt` if necessary after any changes.
- Re-export public APIs in package `__init__.py` files when modules are added or moved.

## Validation

Run the demo after changes:

```bash
python main.py
```
