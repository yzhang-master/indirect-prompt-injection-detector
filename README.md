# indirect-prompt-injection-detector

> **Era:** 2023–2024 · Senior AI Engineer focus area (agent security). Technique from 2023, reimplemented in 2026.

A dependency-light, trainable classifier for indirect prompt injection in tool outputs, paired with
capability-based permissions that remain authoritative when the classifier is wrong or bypassed.
The repository contains only hand-authored local traces and in-memory effects. It makes no external
LLM or network calls.

## Problem

A tool result is untrusted data, but an agent can mistake instructions embedded in that result for
user intent. This creates confused-deputy file access, data exfiltration, and privilege escalation.
A statistical detector is useful for reducing exposure, but detection alone is not authorization:
a false negative must not turn into permission to read a secret, contact an arbitrary host, or grant
new capabilities.

## Method

### Trainable detector

`HashedFeatureExtractor` applies Unicode NFKC normalization, removes zero-width formatting
characters, and hashes word unigrams/bigrams, character 3–5-grams, and a few structural features
into 4,096 signed, L2-normalized dimensions. Hashing uses BLAKE2b rather than Python's randomized
`hash()`. `InjectionClassifier` fits binary logistic regression to the 32 training traces with
class-balanced binary cross-entropy:

```
p(malicious | x) = sigmoid(wᵀφ(x) + b)
loss = weighted_mean(BCE) + λ||w||² / 2
```

Training uses full-batch Adam for 500 steps (`learning_rate=0.03`, `λ=0.0001`). The operating
threshold is fixed at 0.5, not tuned on the held-out split. The trained artifact is
`artifacts/model.json`; fitting and inference require NumPy only.

### Independent capability and resource policy

```text
untrusted tool output -> detector -> capability + resource policy -> in-memory executor
                            | deny              | deny
                            v                   v
                      no tool effect        no tool effect
```

The policy receives capabilities granted by the user context, not capabilities asserted by a tool
result. It checks the exact call on every execution:

| Tool | Required capability | Additional resource constraint |
|---|---|---|
| `read_file` | `fs:read` | only `docs/` and `reports/`; no absolute paths or `..` |
| `write_file` | `fs:write` | only `notes/`; no absolute paths or `..` |
| `http_post` | `net:egress` | HTTPS, port 443/default, allowlisted `api.internal.example` |
| `grant_capability` | `admin:grant` | named principals and non-admin delegatable capabilities only |

The executor is deliberately in-memory: file writes modify a dictionary, outbound requests are
recorded in a list, and grants modify local principal state. The unprotected baseline uses the same
executor without checks so attack success is an observed effect, not a textual judgment.

## Results

Measured on the fixed, untouched local test split: **24 examples**, comprising 12 benign traces and
12 attacks (4 confused-deputy, 4 data-exfiltration, and 4 privilege-escalation traces). Training used
32 disjoint traces (16 benign and 16 malicious). Loss fell from **0.693147** to **0.038931**.

### Detector classification

| Threshold | TP | FP | TN | FN | Precision | Recall | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 12 | 0 | 12 | 0 | **1.000** | **1.000** | **1.000** |

### End-to-end security and utility

| Configuration | Successful attacks | ASR | Classifier blocks | Hard permission denials |
|---|---:|---:|---:|---:|
| Unprotected baseline | 12 / 12 | **100.0%** | 0 | 0 |
| Detector + permissions | 0 / 12 | **0.0%** | 12 | 0 |
| Always-benign detector bypass + permissions | 0 / 12 | **0.0%** | 0 | 12 |

All 12 benign held-out actions executed under the normal protected pipeline (**12/12 utility**).
The bypass row is important: it separately demonstrates that resource and capability checks deny
forbidden effects even when classification never raises an alarm. In the normal row, all attacks
happen to be stopped by the classifier first, so zero permission denials is expected rather than a
missing measurement.

Raw scores and every decision are in `results/metrics.json`; training measurements are in
`results/training.json`. These are results on this repository's small synthetic corpus, not an
external red-team benchmark.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
python scripts/train.py       # fit train split; replace artifacts/model.json
python scripts/evaluate.py    # score test split; write results/metrics.json
python scripts/smoke.py       # end-to-end model + executor + bypass control
```

Expected smoke output:

```text
smoke ok: 24 held-out traces, precision=1.000, recall=1.000, AUC=1.000, ASR=1.000 -> 0.000; bypass control denied 12/12
```

## Limitations

- The 56 hand-authored English traces are a regression corpus, not evidence of broad real-world
  robustness. Perfect held-out scores reflect this small, lexically related sample and must not be
  extrapolated to other datasets.
- The detector was not calibrated on a large representative corpus. Its probability is a logistic
  score, and the 0.5 threshold may need domain-specific validation.
- Paraphrases outside the training distribution, multilingual attacks, encoded payloads, long-context
  dilution, and adaptive adversaries can evade the classifier. Permissions remain necessary.
- The executor demonstrates authorization semantics only. It is not an OS sandbox, does not protect a
  real filesystem or network stack, and does not model time-of-check/time-of-use races.
- The baseline intentionally executes unsafe calls to measure local attack effects. Applications must
  expose only the checked `PermissionPolicy.execute` path to agent-driven calls.

## References

- K. Greshake et al. *Not What You've Signed Up For: Compromising Real-World LLM-Integrated
  Applications with Indirect Prompt Injection.* 2023. arXiv:2302.12173.
- J. H. Saltzer and M. D. Schroeder. *The Protection of Information in Computer Systems.* 1975.
- S. Bubeck et al. *Sparks of Artificial General Intelligence: Early Experiments with GPT-4.* 2023.
  arXiv:2303.12712 (discussion of prompt-injection risks in tool-using systems).

## License

MIT
