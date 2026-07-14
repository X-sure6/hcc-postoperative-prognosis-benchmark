# GitHub Upload Checklist

## Recommended public names

- Repository: `hcc-postoperative-prognosis-benchmark`
- Main script: `hcc_postoperative_prognosis_benchmark.py`

## Files to commit

```text
hcc_postoperative_prognosis_benchmark.py
README.md
requirements.txt
.gitignore
.gitattributes
GITHUB_UPLOAD_CHECKLIST.md
```

## Files that must remain private

Do not commit:

- patient-level Excel or CSV files;
- fixed-fold assignment files containing patient identifiers;
- TabPFN or other model checkpoints;
- generated prediction files or result directories;
- logs containing local server paths;
- files produced with `--save-direct-identifiers`;
- files produced with `--record-absolute-paths`.

## Before committing

Run:

```bash
git status
```

Confirm that only the intended repository files are listed.

Search the repository for possible sensitive content:

```bash
git grep -n -I -E "住院号|病案号|patient_id|/root/|D:\\|初始治疗时间"
```

The main script may legitimately contain column aliases such as `住院号` and
`初始治疗时间`. Review every other match manually.

## Commit commands

```bash
git add hcc_postoperative_prognosis_benchmark.py README.md requirements.txt .gitignore .gitattributes GITHUB_UPLOAD_CHECKLIST.md
git status
git commit -m "Add HCC postoperative prognosis benchmark pipeline"
git push
```
