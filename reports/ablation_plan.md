# Residual normalizer ablation plan

| ID | Scientific question | Configuration |
|---|---|---|
| A0 | What is the unchanged synthetic-only reference? | Baseline checkpoint without a normalizer |
| A1 | Is insertion of the adapter behavior-preserving? | Identity normalizer, no training |
| A2 | Can appearance adaptation alone help? | Train normalizer; freeze all landmarker parameters |
| A3 | Does limited representation adaptation add value? | Train normalizer, HRNet stage 4, and landmark/visibility heads |
| A4 | Is any A3 gain caused only by landmarker fine-tuning? | Fine-tune the same landmarker layers without a normalizer |
| A5 | Does residual control improve the normalizer-only tradeoff? | A2 plus L1/TV image regularization |
| A6 | Does residual control improve joint fine-tuning? | A3 plus L1/TV image regularization |

A0--A3 are the primary sequence. A4 is the key causal control for any claim
that the normalizer itself contributes. A5--A6 should be run only if visual or
quantitative diagnostics show excessive image changes.
