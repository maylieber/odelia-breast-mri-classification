# Archived experiments

Every configuration other than the project's reference configuration (kept at the project
root as `checkpoints/best_model_run.pt`) — alternative backbones, hyperparameters, data
ablations, the leave-one-institution-out cross-validation, and the two-stage cascade. Kept
for provenance.

- `scripts/` — training/eval scripts for each variant
- `checkpoints/` — trained weights for each variant (`.pt`)

These scripts were originally run from the project root, so their imports
(`from dataset import ...`, `from model import ...`, etc.) and relative CSV paths assume
that location. They are not guaranteed to run as-is from here — copy a script back to the
project root (or add the root to `PYTHONPATH`) to rerun one.
