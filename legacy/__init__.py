"""Legacy / historical code kept for reference.

Contents
--------
Fast_gibbs_old.py       - Original prototype Gibbs sampler (unfinished)
old_model_version.py    - Original monolithic model (q = exp(-2*theta) convention)
update_z_old.py         - Pre-matmul _compute_all_disagreements / _update_z methods
old_moves.py            - MH block moves (split-merge, transfer, swap-shift, PY-prior)
old_initialization.py   - Borda-threshold block initialisation helpers

None of these files are imported by the active model code.  They exist solely
so that old algorithmic ideas can be recovered if needed.
"""
