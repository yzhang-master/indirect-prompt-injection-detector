# indirect-prompt-injection-detector

> **Era:** 2023–2024 · Senior AI Engineer focus area (agent security). Technique from 2023, reimplemented in 2026.

A local, trainable detector for indirect prompt injection in tool outputs, paired with capability-based tool permissions that do not depend on the detector. All examples and evaluations are local; no external LLM is called.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
python scripts/smoke.py
```

## License

MIT
