# Public GitHub upload checklist

Before pushing:

- [ ] `git status` contains code/docs/tests only.
- [ ] No patient-level `.xlsx/.xls/.csv/.parquet` files are staged.
- [ ] No fixed-fold file with `sample_id` is staged.
- [ ] No TabPFN checkpoint or other model weights are staged.
- [ ] No patient-level OOF prediction or SHAP `.npy` files are staged.
- [ ] No absolute private server/workstation paths were added to documentation.
- [ ] `PYTHONPATH=. python -m unittest discover -s tests -v` passes.
- [ ] `python -m py_compile hcc_postoperative_prognosis_benchmark.py cgh_supplementary_analyses.py tools/*.py` passes.
- [ ] Private inputs use the exact OS/TTR endpoint names required by the public input contract.
- [ ] For a full GPU reproduction, checkpoint SHA256 matches the value in `REPRODUCIBILITY.md`.
