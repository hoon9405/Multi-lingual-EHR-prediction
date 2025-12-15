import logging
import os
import numpy as np
import pandas as pd

from ehrs import EHR, register_ehr

logger = logging.getLogger(__name__)


@register_ehr("sicdb")
class SICDB(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ehr_name = "sicdb"

        if self.data_dir is None:
            raise ValueError("Please provide --data pointing to the SICDB folder.")

        logger.info("Data directory is set to {}".format(self.data_dir))

        if self.ext is None:
            self.ext = self.infer_data_extension()

        # file names
        self._icustay_fname = "cases" + self.ext
        self._unitlog_fname = "unitlog" + self.ext
        self._references_fname = "d_references" + self.ext

        # task tables (all timestamps are in seconds from ICU admission)
        self.tables = [
            {
                "fname": "laboratory" + self.ext,
                "timestamp": "Offset",
                "timeoffsetunit": "sec",
                "exclude": [
                    "id",
                    "PatientID",
                    "LaboratoryType",
                ],
                "code": ["DrugID"],
                "desc": [self._references_fname],
                "desc_key": [
                    [
                        "ReferenceValue",
                        "ReferenceName",
                        "ReferenceDescription",
                        "ReferenceUnit",
                    ]
                ],
                "desc_code_col": ["ReferenceGlobalID"],
                "rename_map": [{"ReferenceValue": "DrugID"}],
            },
            {
                "fname": "medication" + self.ext,
                "timestamp": "Offset",
                "timeoffsetunit": "sec",
                "exclude": [
                    "id",
                    "PatientID",
                    "OffsetDrugEnd",
                    "IsSingleDose",
                ],
                "code": ["DrugID"],
                "desc": [self._references_fname],
                "desc_key": [
                    [
                        "ReferenceValue",
                        "ReferenceName",
                        "ReferenceDescription",
                        "ReferenceUnit",
                    ]
                ],
                "desc_code_col": ["ReferenceGlobalID"],
                "rename_map": [{"ReferenceValue": "DrugID"}],
            },
            {
                "fname": "data_float_h" + self.ext,
                "timestamp": "Offset",
                "timeoffsetunit": "sec",
                "exclude": [
                    "id",
                    "rawdata",
                ],
                "code": ["DataID"],
                "desc": [self._references_fname],
                "desc_key": [
                    [
                        "ReferenceValue",
                        "ReferenceName",
                        "ReferenceDescription",
                        "ReferenceUnit",
                    ]
                ],
                "desc_code_col": ["ReferenceGlobalID"],
                "rename_map": [{"ReferenceValue": "DataID"}],
            },
            {
                "fname": "data_range" + self.ext,
                "timestamp": "Offset",
                "timeoffsetunit": "sec",
                "exclude": [
                    "id",
                    "OffsetEnd",
                    "Data",
                ],
                "code": ["DataID"],
                "desc": [self._references_fname],
                "desc_key": [
                    [
                        "ReferenceValue",
                        "ReferenceName",
                        "ReferenceDescription",
                        "ReferenceUnit",
                    ]
                ],
                "desc_code_col": ["ReferenceGlobalID"],
                "rename_map": [{"ReferenceValue": "DataID"}],
            },
        ]

        # keys
        self._icustay_key = "CaseID"
        self._hadm_key = "CaseID"
        self._patient_key = "PatientID"
        self._determine_first_icu = "INTIME"

        self.task_itemids = {
            "creatinine": {
                "fname": "laboratory" + self.ext,
                "timestamp": "Offset",
                "exclude": ["id", "PatientID", "LaboratoryType"],
                "code": ["LaboratoryID"],
                "value": ["LaboratoryValue"],
                "itemid": [339, 367, 368, 369, 370, 380],
            },
            "platelets": {
                "fname": "laboratory" + self.ext,
                "timestamp": "Offset",
                "exclude": ["id", "PatientID", "LaboratoryType"],
                "code": ["LaboratoryID"],
                "value": ["LaboratoryValue"],
                "itemid": [314, 315],
            },
            "wbc": {
                "fname": "laboratory" + self.ext,
                "timestamp": "Offset",
                "exclude": ["id", "PatientID", "LaboratoryType"],
                "code": ["LaboratoryID"],
                "value": ["LaboratoryValue"],
                "itemid": [301],
            },
            "hb": {
                "fname": "laboratory" + self.ext,
                "timestamp": "Offset",
                "exclude": ["id", "PatientID", "LaboratoryType"],
                "code": ["LaboratoryID"],
                "value": ["LaboratoryValue"],
                "itemid": [139, 658],
            },
            "bicarbonate": {
                "fname": "laboratory" + self.ext,
                "timestamp": "Offset",
                "exclude": ["id", "PatientID", "LaboratoryType"],
                "code": ["LaboratoryID"],
                "value": ["LaboratoryValue"],
                "itemid": [666, 667],
            },
            "sodium": {
                "fname": "laboratory" + self.ext,
                "timestamp": "Offset",
                "exclude": ["id", "PatientID", "LaboratoryType"],
                "code": ["LaboratoryID"],
                "value": ["LaboratoryValue"],
                "itemid": [469, 536],
            },
            "urine": {
                "fname": "data_float_h" + self.ext,
                "timestamp": "Offset",
                "exclude": ["id", "PatientID", "cnt", "rawdata"],
                "code": ["DataID"],
                "value": ["Val"],
                "itemid": [725],
            },
            "AKI": {
                "fname": "laboratory" + self.ext,
                "timestamp": "Offset",
                "exclude": ["id", "PatientID", "LaboratoryType"],
                "code": ["LaboratoryID"],
                "value": ["LaboratoryValue"],
                "itemid": [339, 367, 368, 369, 370, 380],
            },
        }

        self._dialysis_dataids = [723, 730, 731, 732, 2022]

    def _load_references(self):
        ref_path = os.path.join(self.data_dir, self._references_fname)
        if not os.path.exists(ref_path):
            logger.warning("d_references file not found at {}".format(ref_path))
            return pd.DataFrame()
        return pd.read_csv(ref_path)

    def _icu_unit_ids(self, references):
        """
        Derive ICU/IMCU HospitalUnit IDs from d_references; fallback to all HospitalUnit IDs.
        """
        icu_unit_values = {"INIC", "INBD", "INID", "CWIN", "INCV", "CWCV"}
        if references.empty:
            return None

        hosp_units = references[references["ReferenceName"] == "HospitalUnit"]
        value_to_id = dict(
            zip(hosp_units["ReferenceValue"], hosp_units["ReferenceGlobalID"])
        )
        icu_ids = {value_to_id[v] for v in icu_unit_values if v in value_to_id}
        if icu_ids:
            return icu_ids

        # fallback: treat every HospitalUnit code as ICU/IMCU
        return set(hosp_units["ReferenceGlobalID"])

    def _compute_icu_block(self, case_row, log_df, icu_units):
        """
        Compute first ICU block [icu_in, icu_out) in seconds.
        """
        icu_in = None
        icu_out = None

        if (icu_units is None or len(icu_units) == 0) and log_df is not None and not log_df.empty:
            # fallback: treat all observed HospitalUnit codes as ICU/IMCU
            icu_units = set(log_df["HospitalUnit"].astype(str).unique())

        if log_df is not None and not log_df.empty and icu_units:
            log_df = log_df.sort_values("Offset")
            icu_mask = log_df["HospitalUnit"].astype(str).isin(icu_units)
            icu_logs = log_df[icu_mask]
            if not icu_logs.empty:
                icu_in = int(float(icu_logs.iloc[0]["Offset"]))
                after_in = log_df[log_df["Offset"] >= icu_in]
                non_icu = after_in[~after_in["HospitalUnit"].astype(str).isin(icu_units)]
                if not non_icu.empty:
                    icu_out = int(float(non_icu.iloc[0]["Offset"]))

        if icu_in is None:
            icu_in = int(float(case_row.get("ICUOffset", 0) or 0))

        if icu_out is None:
            icu_out = icu_in + int(float(case_row.get("TimeOfStay", 0) or 0))

        return icu_in, icu_out

    def make_compatible(self, icustays, spark):
        # icustays is cases.csv loaded by caller
        cases = icustays.copy()

        references = self._load_references()
        icu_units = self._icu_unit_ids(references)
        unitlog_path = os.path.join(self.data_dir, self._unitlog_fname)
        if os.path.exists(unitlog_path):
            unitlog = pd.read_csv(unitlog_path)
            unitlog["CaseID"] = unitlog["CaseID"].astype(str)
            unitlog_groups = {cid: df for cid, df in unitlog.groupby("CaseID")}
        else:
            logger.warning("unitlog file not found; falling back to ICUOffset/TimeOfStay")
            unitlog_groups = dict()

        icu_in_list = []
        icu_out_list = []
        drop_idx = []

        for idx, row in cases.iterrows():
            case_id = str(row["CaseID"])
            logs = unitlog_groups.get(case_id)
            icu_in, icu_out = self._compute_icu_block(row, logs, icu_units)
            if icu_out <= icu_in:
                drop_idx.append(idx)
                continue
            icu_in_list.append(icu_in)
            icu_out_list.append(icu_out)

        # align lengths
        kept_cases = cases.drop(index=drop_idx).copy()
        kept_cases["icu_in_sec"] = icu_in_list
        kept_cases["icu_out_sec"] = icu_out_list

        kept_cases["INTIME"] = kept_cases["icu_in_sec"] / 60.0
        kept_cases["ADMITTIME"] = 0
        kept_cases["LOS"] = (kept_cases["icu_out_sec"] - kept_cases["icu_in_sec"]) / 86400.0

        # DEATHTIME (minutes)
        def compute_deathtime(row):
            offset_death = row.get("OffsetOfDeath", "")
            try:
                if pd.notnull(offset_death) and str(offset_death).strip() != "":
                    return float(offset_death) / 60.0
            except Exception:
                pass

            if str(row.get("HospitalDischargeType", "")) == "2028":
                return float(row.get("TimeOfStay", 0) or 0) / 60.0
            if str(row.get("DischargeState", "")) == "2215":
                return float(row.get("TimeOfStay", 0) or 0) / 60.0
            return np.nan

        kept_cases["DEATHTIME"] = kept_cases.apply(compute_deathtime, axis=1)

        # AGE & weight
        kept_cases["AGE"] = pd.to_numeric(kept_cases["AgeOnAdmission"], errors="coerce")
        kept_cases.loc[kept_cases["AGE"] > 90, "AGE"] = 90
        kept_cases["weight"] = pd.to_numeric(
            kept_cases.get("WeightOnAdmission", np.nan), errors="coerce"
        )
        # SICDB weight은 g 단위로 저장되어 있으므로 kg로 변환
        kept_cases.loc[kept_cases["weight"] > 500, "weight"] = (
            kept_cases.loc[kept_cases["weight"] > 500, "weight"] / 1000.0
        )

        if kept_cases["weight"].notnull().any():
            default_weight = kept_cases["weight"].median()
        else:
            default_weight = 72.5
        kept_cases["weight"] = kept_cases["weight"].fillna(default_weight)
        kept_cases.loc[kept_cases["weight"] <= 0, "weight"] = default_weight

        # drop helper columns
        kept_cases = kept_cases.drop(columns=["icu_in_sec", "icu_out_sec"], errors="ignore")

        return kept_cases

    def prepare_tasks(self, cohorts, spark=None, cached=False):
        labeled = super().prepare_tasks(cohorts, spark, cached)
        if cached:
            return labeled
        labeled = labeled.loc[:, ~labeled.columns.duplicated()].copy()

        clinical_tasks = [
            "creatinine",
            "platelets",
            "wbc",
            "hb",
            "bicarbonate",
            "sodium",
            "urine",
            "AKI",
        ]

        for task in clinical_tasks:
            horizons = self.__getattribute__(task)
            if not horizons:
                continue
            labeled = self.clinical_task(labeled, task, [int(h) for h in horizons])

        return labeled

    def _load_task_table(self, meta):
        fname = os.path.join(self.data_dir, meta["fname"])
        usecols = {self.icustay_key, meta["timestamp"], meta["code"][0]}
        if "value" in meta:
            usecols.add(meta["value"][0])
        table = pd.read_csv(fname, usecols=list(usecols))
        for col in list(meta.get("exclude", [])):
            if col in table.columns and col not in usecols:
                table = table.drop(columns=col)
        return table

    def _normalize_units(self, df, task, value_col):
        # SICDB laboratory items already match mg/dL, g/dL, mmol/L, and G/L,
        # but urine (ml) needs to convert to ml/kg/hr later via weight.
        return df

    def _ensure_dialysis_lookup(self, case_ids):
        if not hasattr(self, "_dialysis_lookup"):
            self._dialysis_lookup = dict()

        missing = {int(cid) for cid in case_ids if int(cid) not in self._dialysis_lookup}
        if not missing:
            return

        path = os.path.join(self.data_dir, "data_float_h" + self.ext)
        usecols = [self.icustay_key, "DataID", "Offset"]
        chunk_iter = pd.read_csv(path, usecols=usecols, chunksize=500000)
        for chunk in chunk_iter:
            chunk[self.icustay_key] = pd.to_numeric(chunk[self.icustay_key], errors="coerce").astype("Int64")
            chunk = chunk[chunk[self.icustay_key].isin(missing)]
            if chunk.empty:
                continue
            chunk = chunk[chunk["DataID"].isin(self._dialysis_dataids)]
            if chunk.empty:
                continue
            earliest = (
                chunk.groupby(self.icustay_key)["Offset"].min().astype(float) / 60.0
            )
            for cid, val in earliest.items():
                self._dialysis_lookup[int(cid)] = float(val)
                missing.discard(int(cid))
            if not missing:
                break

        for cid in list(missing):
            self._dialysis_lookup[int(cid)] = None

    def dialysis_filter(self, df, timestamp_col):
        case_ids = df[self.icustay_key].unique()
        if len(case_ids) == 0:
            return df
        self._ensure_dialysis_lookup(case_ids)
        lookup = pd.DataFrame(
            {
                self.icustay_key: case_ids,
                "_dialysis_min": [self._dialysis_lookup.get(int(cid)) for cid in case_ids],
            }
        )
        df = df.merge(lookup, on=self.icustay_key, how="left")
        df = df[
            df["_dialysis_min"].isna() | (df["_dialysis_min"] > df[timestamp_col])
        ]
        return df.drop(columns="_dialysis_min")

    def clinical_task(self, cohorts, task, horizons):
        meta = self.task_itemids[task]
        timestamp = meta["timestamp"]
        code_col = meta["code"][0]
        value_col = meta["value"][0]
        itemids = meta["itemid"]

        table = self._load_task_table(meta)
        table[code_col] = pd.to_numeric(table[code_col], errors="coerce")
        table = table[table[code_col].isin(itemids)]
        table[value_col] = pd.to_numeric(table[value_col], errors="coerce")
        table = table.dropna(subset=[value_col, timestamp])

        table[timestamp] = pd.to_numeric(table[timestamp], errors="coerce") / 60.0
        table = table.rename(columns={timestamp: "EVENT_TIME"})
        table = self._normalize_units(table, task, value_col)

        cohort_view = cohorts.loc[:, ~cohorts.columns.duplicated()].copy()
        needed_cols = [self.icustay_key, "INTIME", "ADMITTIME"]
        if "weight" in cohorts.columns:
            needed_cols.append("weight")
        merge = table.merge(
            cohort_view[needed_cols],
            on=self.icustay_key,
            how="inner",
        )
        merge["EVENT_TIME"] = merge["EVENT_TIME"] + merge["INTIME"]
        merge = merge.dropna(subset=["EVENT_TIME"])

        if task in ["creatinine", "urine", "AKI"]:
            merge = self.dialysis_filter(merge, "EVENT_TIME")

        merge = merge[merge["EVENT_TIME"] >= merge["INTIME"] + self.pred_size * 60]

        baseline_lookup = None
        if task == "AKI":
            baseline_window = merge[
                (merge["EVENT_TIME"] >= merge["INTIME"] - 7 * 24 * 60)
                & (merge["EVENT_TIME"] < merge["INTIME"] + 12 * 60)
            ]
            baseline_lookup = (
                baseline_window.groupby(self.icustay_key)[value_col].min().to_dict()
            )
            icu_events = merge[merge["EVENT_TIME"] >= merge["INTIME"]]
            icu_events = icu_events.sort_values(["EVENT_TIME"])
            first_measure = icu_events.groupby(self.icustay_key)[value_col].first()
            for cid, val in first_measure.items():
                baseline_lookup[cid] = min(
                    baseline_lookup.get(cid, val), val
                ) if cid in baseline_lookup else val

        for horizon in horizons:
            start = merge["INTIME"] + (self.pred_size + (horizon - 1) * 24) * 60
            end = merge["INTIME"] + (self.pred_size + horizon * 24) * 60
            mask = (merge["EVENT_TIME"] >= start) & (merge["EVENT_TIME"] < end)
            horizon_df = merge[mask]
            if horizon_df.empty:
                continue

            agg_col = value_col
            if task == "urine":
                if "weight" not in horizon_df.columns:
                    continue
                horizon_df = horizon_df[horizon_df["weight"] > 0]
                horizon_df = horizon_df.assign(
                    _URINE_RATE=horizon_df[value_col] / horizon_df["weight"]
                )
                agg_col = "_URINE_RATE"

            horizon_agg = (
                horizon_df.groupby(self.icustay_key)[agg_col]
                .median()
                .to_frame("value")
            )

            if task == "AKI" and baseline_lookup is not None:
                horizon_agg["baseline"] = horizon_agg.index.map(
                    lambda cid: baseline_lookup.get(cid)
                )

            task_name = f"{task}_{horizon}"
            if task == "platelets":
                horizon_agg[task_name] = pd.cut(
                    horizon_agg["value"],
                    bins=[-float("inf"), 20, 50, 100, 150, float("inf")],
                    labels=[4, 3, 2, 1, 0],
                    include_lowest=True,
                ).astype(float)
            elif task == "creatinine":
                horizon_agg[task_name] = pd.cut(
                    horizon_agg["value"],
                    bins=[-float("inf"), 1.2, 2.0, 3.5, 5.0, float("inf")],
                    labels=[0, 1, 2, 3, 4],
                    include_lowest=True,
                ).astype(float)
            elif task == "wbc":
                horizon_agg[task_name] = pd.cut(
                    horizon_agg["value"],
                    bins=[-float("inf"), 4, 12, float("inf")],
                    labels=[0, 1, 2],
                    include_lowest=True,
                ).astype(float)
            elif task == "hb":
                horizon_agg[task_name] = pd.cut(
                    horizon_agg["value"],
                    bins=[-float("inf"), 8, 10, 12, float("inf")],
                    labels=[0, 1, 2, 3],
                    include_lowest=True,
                ).astype(float)
            elif task == "bicarbonate":
                horizon_agg[task_name] = pd.cut(
                    horizon_agg["value"],
                    bins=[-float("inf"), 22, 29, float("inf")],
                    labels=[0, 1, 2],
                    include_lowest=True,
                ).astype(float)
            elif task == "sodium":
                horizon_agg[task_name] = pd.cut(
                    horizon_agg["value"],
                    bins=[-float("inf"), 135, 145, float("inf")],
                    labels=[0, 1, 2],
                    include_lowest=True,
                ).astype(float)
            elif task == "urine":
                horizon_agg[task_name] = pd.cut(
                    horizon_agg["value"],
                    bins=[-float("inf"), 0.05, 0.1, 0.3, 0.5, float("inf")],
                    labels=[4, 3, 2, 1, 0],
                    include_lowest=True,
                ).astype(float)
            elif task == "AKI":
                horizon_agg = horizon_agg.dropna(subset=["baseline"])
                horizon_agg["rel_change"] = (
                    (horizon_agg["value"] - horizon_agg["baseline"])
                    / horizon_agg["baseline"]
                )
                horizon_agg[task_name] = pd.cut(
                    horizon_agg["rel_change"],
                    bins=[-float("inf"), 0.3, 1.5, 2.0, 3.0, float("inf")],
                    labels=[0, 1, 2, 3, 4],
                    include_lowest=True,
                ).astype(float)
                horizon_agg = horizon_agg.drop(columns=["rel_change"])

            cohorts = cohorts.merge(
                horizon_agg[[task_name]],
                left_on=self.icustay_key,
                right_index=True,
                how="left",
            )

            if task == "AKI" and f"urine_{horizon}" in cohorts.columns:
                cohorts[task_name] = cohorts[[task_name, f"urine_{horizon}"]].max(
                    axis=1, skipna=False
                )

        return cohorts
