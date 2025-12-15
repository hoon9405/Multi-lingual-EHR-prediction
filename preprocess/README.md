# Integrated-EHR-Pipeline
- Refined EHR preprocessing pipeline based on GenHPF

## Requirements
- `python>=3.9`, `Java>=8`
- Recommended: `transformers==4.29.1`
```
pip install numpy pandas tqdm treelib transformers==4.29.1 pyspark
```

## How to run preprocessing
```
python main.py \
  --ehr mimiciv \
  --data /path/to/mimiciv \
  --dest /path/to/outputs \
  --obs_size 48 --pred_size 48 \
  --num_threads 16
```
- Replace `--data` and `--dest` with your paths. Physionet datasets require valid credentials.
- All predefined tasks (e.g., mortality, LOS, readmission, etc.) are labeled automatically; no separate labeling step is needed. Task toggles follow defaults in `preprocess/main.py` and can be overridden via CLI flags.
  - To run a subset of tasks, disable others via flags (e.g., `--no-readmission` style flags are not defined; instead, set only what you need: `--mortality 1 2` and omit `--readmission`, `--diagnosis`, etc., by leaving them at defaults or customizing the parser as needed).
  - Label stats are saved to `dest/label_stats_{ehr}.csv`; set `--verify_labeling` to print and exit after labeling.

## Supported EHRs
- mimiciii, mimiciv, eicu, umcdb, hirid, sicdb, ehrshot, nwicu (and any other class registered in `EHR_REGISTRY`). Select with `--ehr`.

## Language analysis
Collect word counts and run language detection using the same EHR definitions:
```
python preprocess/lang_analysis.py \
  --ehr mimiciv \
  --data /path/to/mimiciv \
  --dest /path/to/outputs \
  --row-limit 10000        # optional sampling
# add --skip-detect to only save word counts
```
- Outputs:
  - Word counts per table/column: `dest/lang_counts/{ehr}_{table}.json`
  - Language distribution summary: `dest/lang_counts/{ehr}_lang_distribution.json`
  - Detected languages with per-word stats: `dest/{ehr}_lang_dict.pkl`

## Cache
- Use `--cache` to reuse previously saved cohorts/labels from `~/.cache/ehr/{ehr}`. This speeds up reruns but may ignore changed CLI flags (e.g., `--first_icu`, task toggles). Remove or avoid `--cache` when changing preprocessing options.
