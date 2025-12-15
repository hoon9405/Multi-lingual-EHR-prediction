import argparse
import glob
import importlib.util
import json
import os
import pickle
from pathlib import Path

import pandas as pd
from langid.langid import LanguageIdentifier, model
from tqdm import tqdm

from utils import detect_language, get_word_freq
from ehrs import EHR_REGISTRY


def _load_preprocess_parser():
    """Load the preprocessing parser without importing the repository root main module."""
    main_path = Path(__file__).with_name("main.py")
    spec = importlib.util.spec_from_file_location("preprocess_main", main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_parser()


def get_parser():
    parser = _load_preprocess_parser()
    parser.description = "Collect word counts and run language detection using EHR definitions."
    parser.add_argument(
        "--skip-detect",
        action="store_true",
        help="Only collect word counts; skip language detection and pickle export.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Optional row limit per table for faster counting (e.g., smoke tests).",
    )
    return parser


def _table_name(fname, ext):
    if ext:
        fname = fname.split(ext)[0]
    return os.path.basename(fname)


def _load_table_df(ehr, table, row_limit=None):
    fname = table["fname"]
    table_path = os.path.join(ehr.data_dir, fname)

    if ehr.ehr_name == "umcdb":
        df = pd.read_csv(table_path, encoding="ISO-8859-1", nrows=row_limit)
    elif ehr.ehr_name == "hirid" and os.path.isdir(table_path):
        csv_files = [
            os.path.join(table_path, f)
            for f in os.listdir(table_path)
            if f.endswith(".csv")
        ]
        rows_read = 0
        all_dataframes = []
        for file in csv_files:
            remaining = None if row_limit is None else max(row_limit - rows_read, 0)
            if remaining == 0:
                break
            df_part = pd.read_csv(file, nrows=remaining if remaining else None)
            all_dataframes.append(df_part)
            rows_read += len(df_part)
            if row_limit and rows_read >= row_limit:
                break
        df = pd.concat(all_dataframes, ignore_index=True)
    else:
        df = pd.read_csv(table_path, nrows=row_limit)

    if "code" in table:
        for code_idx, code in enumerate(table["code"]):
            desc = table["desc"][code_idx]
            mapping_table = pd.read_csv(os.path.join(ehr.data_dir, desc))

            if "desc_filter_col" in table:
                mapping_table = mapping_table[
                    mapping_table[table["desc_filter_col"][code_idx]]
                    == table["desc_filter_val"][code_idx]
                ]

            mapping_table = mapping_table.rename(
                columns={table["desc_code_col"][code_idx]: code}
            )
            mapping_table = mapping_table[[code] + table["desc_key"][code_idx]]

            df = df.merge(mapping_table, on=code, how="left")
            df = df.drop(columns=[code])

            for k, v in table["rename_map"][code_idx].items():
                df = df.rename(columns={k: v})

    return df


def collect_word_counts(ehr, row_limit=None):
    lang_counts_dir = os.path.join(ehr.dest, "lang_counts")
    os.makedirs(lang_counts_dir, exist_ok=True)

    for table in ehr.tables:
        table_name = _table_name(table["fname"], ehr.ext)
        timestamp_key = table["timestamp"]
        excludes = set(table["exclude"])
        for key in [ehr.icustay_key, ehr.hadm_key, timestamp_key]:
            if key:
                excludes.add(key)

        print(f"Collecting word counts for table: {table_name}")
        df = _load_table_df(ehr, table, row_limit=row_limit)

        table_word_counts = {}
        for col in tqdm(df.columns, desc=f"{table_name} columns"):
            if col in excludes:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series.dtype):
                continue
            word_freq = get_word_freq(series)
            if word_freq:
                table_word_counts[col] = word_freq

        out_path = os.path.join(lang_counts_dir, f"{ehr.ehr_name}_{table_name}.json")
        with open(out_path, "w") as f:
            json.dump(table_word_counts, f)

    return lang_counts_dir


def detect_languages(lang_counts_dir, dest=None, ehr_name=None):
    ident = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    data_dict = {}
    lang_summary = {"overall": {}}

    for file in glob.glob(os.path.join(lang_counts_dir, "*.json")):
        base_name = os.path.basename(file).replace(".json", "")
        # skip previously generated summary files
        if base_name.endswith("lang_distribution"):
            continue
        if ehr_name and base_name.startswith(f"{ehr_name}_"):
            table_name = base_name[len(ehr_name) + 1 :]
        else:
            table_name = base_name
        print(f"Running language detection for {table_name}...")

        with open(file, "r") as f:
            table_word_counts = json.load(f)

        table_dict = {}
        table_lang_counts = {}
        column_lang_counts = {}
        for col, counts in tqdm(table_word_counts.items()):
            counts = {k: int(v) for k, v in counts.items()}
            detected = detect_language(counts, ident)
            table_dict[col] = detected

            col_counts = {}
            for _, (lang, _, cnt) in detected.items():
                col_counts[lang] = col_counts.get(lang, 0) + cnt
                table_lang_counts[lang] = table_lang_counts.get(lang, 0) + cnt
                lang_summary["overall"][lang] = lang_summary["overall"].get(lang, 0) + cnt
            column_lang_counts[col] = col_counts

        data_dict[table_name] = table_dict
        lang_summary[table_name] = {
            "table": table_lang_counts,
            "columns": column_lang_counts,
        }

    if dest:
        os.makedirs(os.path.join(dest, "lang_counts"), exist_ok=True)
        prefix = f"{ehr_name}_" if ehr_name else ""
        summary_path = os.path.join(dest, "lang_counts", f"{prefix}lang_distribution.json")
        with open(summary_path, "w") as f:
            json.dump(lang_summary, f)
        print(f"Language distribution summary saved to {summary_path}")

    return data_dict, lang_summary


def main(args):
    ehr = EHR_REGISTRY[args.ehr](args)
    os.makedirs(ehr.dest, exist_ok=True)

    lang_counts_dir = collect_word_counts(ehr, row_limit=args.row_limit)
    if args.skip_detect:
        print(f"Word counts saved under {lang_counts_dir}. Skipping language detection.")
        return

    data_dict, _ = detect_languages(lang_counts_dir, dest=ehr.dest, ehr_name=ehr.ehr_name)
    output_path = os.path.join(ehr.dest, f"{ehr.ehr_name}_lang_dict.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"Language analysis results saved to {output_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
