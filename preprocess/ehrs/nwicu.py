import glob
import logging
import os
from typing import List

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from ehrs import EHR, register_ehr

logger = logging.getLogger(__name__)


@register_ehr("nwicu")
class NwICU(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ehr_name = "nwicu"

        if self.data_dir is None:
            raise ValueError("Please provide --data pointing to the NWICU root directory.")

        if self.ext is None:
            self.ext = self.infer_data_extension()

        self._icustay_fname = os.path.join("nw_icu", "icustays" + self.ext)
        self._patient_fname = os.path.join("nw_hosp", "patients" + self.ext)
        self._admission_fname = os.path.join("nw_hosp", "admissions" + self.ext)
        self._diagnosis_fname = os.path.join("nw_hosp", "diagnoses_icd" + self.ext)

        self._icustay_key = "stay_id"
        self._hadm_key = "hadm_id"
        self._patient_key = "subject_id"
        self._determine_first_icu = "intime"

        # Minimal tables for event extraction (extend like mimiciv if needed)
        self.tables = [
            {
                "fname": os.path.join("nw_hosp", "labevents" + self.ext),
                "timestamp": "charttime",
                "timeoffsetunit": "abs",
                "exclude": [
                    "labevent_id",
                    "storetime",
                    "valuemgdl",
                    "dilution",
                    "flag",
                    "priority",
                    "comments",
                ],
                "code": ["itemid"],
                "desc": [os.path.join("nw_hosp", "d_labitems" + self.ext)],
                "desc_key": [["label"]],
                "desc_code_col": ["itemid"],
                "rename_map": [{"label": "itemid"}],
            },
            {
                "fname": os.path.join("nw_icu", "chartevents" + self.ext),
                "timestamp": "charttime",
                "timeoffsetunit": "abs",
                "exclude": ["storetime", "cgid", "warning"],
                "code": ["itemid"],
                "desc": [os.path.join("nw_icu", "d_items" + self.ext)],
                "desc_key": [["label"]],
                "desc_code_col": ["itemid"],
                "rename_map": [{"label": "itemid"}],
            },
            {
                "fname": os.path.join("nw_icu", "procedureevents" + self.ext),
                "timestamp": "starttime",
                "timeoffsetunit": "abs",
                "exclude": [
                    "endtime",
                    "storetime",
                    "cgid",
                    "orderid",
                    "linkorderid",
                    "location",
                    "locationcategory",
                    "ordercategoryname",
                    "ordercategorydescription",
                    "statusdescription",
                    "isopenbag",
                    "continueinnextdept",
                ],
                "code": ["itemid"],
                "desc": [os.path.join("nw_icu", "d_items" + self.ext)],
                "desc_key": [["label"]],
                "desc_code_col": ["itemid"],
                "rename_map": [{"label": "itemid"}],
            },
        ]

        # ItemID mapping from d_labitems/d_items (may need refinement)
        self.task_itemids = {
            "creatinine": {
                "fname": os.path.join("nw_hosp", "labevents" + self.ext),
                "timestamp": "charttime",
                "exclude": ["labevent_id", "storetime", "flag", "priority", "comments"],
                "code": ["itemid"],
                "value": ["valuenum"],
                "itemid": [100002],
            },
            "AKI": {
                # Creatinine 기반 AKI 계산용(동일 itemid 활용)
                "fname": os.path.join("nw_hosp", "labevents" + self.ext),
                "timestamp": "charttime",
                "exclude": ["labevent_id", "storetime", "flag", "priority", "comments"],
                "code": ["itemid"],
                "value": ["valuenum"],
                "itemid": [100002],
            },
            "platelets": {
                "fname": os.path.join("nw_hosp", "labevents" + self.ext),
                "timestamp": "charttime",
                "exclude": ["labevent_id", "storetime", "flag", "priority", "comments"],
                "code": ["itemid"],
                "value": ["valuenum"],
                "itemid": [100014],
            },
            "wbc": {
                "fname": os.path.join("nw_hosp", "labevents" + self.ext),
                "timestamp": "charttime",
                "exclude": ["labevent_id", "storetime", "flag", "priority", "comments"],
                "code": ["itemid"],
                "value": ["valuenum"],
                "itemid": [100016, 100083],
            },
            "hb": {
                "fname": os.path.join("nw_hosp", "labevents" + self.ext),
                "timestamp": "charttime",
                "exclude": ["labevent_id", "storetime", "flag", "priority", "comments"],
                "code": ["itemid"],
                "value": ["valuenum"],
                "itemid": [100007],
            },
            "bicarbonate": {
                "fname": os.path.join("nw_hosp", "labevents" + self.ext),
                "timestamp": "charttime",
                "exclude": ["labevent_id", "storetime", "flag", "priority", "comments"],
                "code": ["itemid"],
                "value": ["valuenum"],
                "itemid": [100013],
            },
            "sodium": {
                "fname": os.path.join("nw_hosp", "labevents" + self.ext),
                "timestamp": "charttime",
                "exclude": ["labevent_id", "storetime", "flag", "priority", "comments"],
                "code": ["itemid"],
                "value": ["valuenum"],
                "itemid": [100010, 100050],
            },
            # urine 미지원: 존재하지 않는 경우 -1로 채움
            "urine": {
                "fname": None,
                "timestamp": None,
                "exclude": [],
                "code": [None],
                "value": [None],
                "itemid": [],
            },
            "dialysis": {
                "fname": os.path.join("nw_icu", "procedureevents" + self.ext),
                "timestamp": "starttime",
                "exclude": ["endtime", "storetime"],
                "code": ["itemid"],
                "value": ["value"],
                "itemid": [704890, 724671, 772042, 793064, 798351],
            },
        }

        self.weight_itemid = 326531  # chartevents, ounce

    def infer_data_extension(self) -> str:
        if glob.glob(os.path.join(self.data_dir, "**", "*.csv.gz"), recursive=True):
            return ".csv.gz"
        elif glob.glob(os.path.join(self.data_dir, "**", "*.csv"), recursive=True):
            return ".csv"
        raise AssertionError("Cannot infer data extension for NWICU.")

    def make_compatible(self, icustays: pd.DataFrame, spark):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patient_fname))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admission_fname))

        icustays = icustays.rename(
            columns={
                "stay_id": self.icustay_key,
                "intime": "INTIME_RAW",
                "outtime": "OUTTIME_RAW",
                "los": "LOS",
            }
        )
        icustays["INTIME_RAW"] = pd.to_datetime(icustays["INTIME_RAW"])
        icustays["OUTTIME_RAW"] = pd.to_datetime(icustays["OUTTIME_RAW"])

        admissions = admissions.rename(
            columns={"admittime": "ADMITTIME", "deathtime": "DEATHTIME"}
        )
        admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])
        admissions["DEATHTIME"] = pd.to_datetime(admissions["DEATHTIME"])

        icustays = icustays.merge(
            admissions[[self.hadm_key, "ADMITTIME", "DEATHTIME"]],
            on=self.hadm_key,
            how="left",
        )

        icustays = icustays.merge(patients, on=self.patient_key, how="left")
        icustays["INTIME"] = (
            icustays["INTIME_RAW"] - icustays["ADMITTIME"]
        ).dt.total_seconds() // 60
        icustays["LOS"] = icustays["LOS"]

        icustays["AGE"] = (
            icustays["INTIME_RAW"].dt.year
            - icustays["anchor_year"]
            + icustays["anchor_age"]
        )

        # Convert DEATHTIME to minutes from admission for downstream mortality calc
        icustays["DEATHTIME"] = (
            icustays["DEATHTIME"] - icustays["ADMITTIME"]
        ).dt.total_seconds() // 60

        # Weight from chartevents (ounces -> kg)
        chartevents = spark.read.csv(
            os.path.join(self.data_dir, "nw_icu", "chartevents" + self.ext), header=True
        )
        weights = (
            chartevents.filter(F.col("itemid") == self.weight_itemid)
            .select(self.icustay_key, "valuenum", "valueuom")
            .withColumn("valuenum", F.col("valuenum").cast("double"))
        )
        weights = weights.withColumn(
            "weight",
            F.when(F.lower(F.col("valueuom")) == "ounce", F.col("valuenum") * 0.0283495).otherwise(
                F.col("valuenum")
            ),
        )
        weights = weights.groupBy(self.icustay_key).agg(F.avg("weight").alias("weight")).toPandas()
        weights[self.icustay_key] = weights[self.icustay_key].astype("int64")
        icustays[self.icustay_key] = icustays[self.icustay_key].astype("int64")
        icustays = icustays.merge(weights, on=self.icustay_key, how="left")

        default_by_gender = {"M": 80.0, "F": 65.0}
        if "gender" in icustays.columns:
            for g, default in default_by_gender.items():
                mask = (icustays["gender"] == g) & icustays["weight"].isna()
                icustays.loc[mask, "weight"] = default
        icustays["weight"] = icustays["weight"].fillna(72.5)

        icustays["readmission"] = None
        return icustays

    def prepare_tasks(self, cohorts, spark, cached=False):
        labeled_cohorts = super().prepare_tasks(cohorts, spark, cached)
        if cached:
            return labeled_cohorts

        logger.info("Start labeling cohorts for clinical task prediction.")
        labeled_cohorts = spark.createDataFrame(labeled_cohorts)
        for clinical_task in [
            "creatinine",
            "platelets",
            "wbc",
            "hb",
            "bicarbonate",
            "sodium",
            "urine",
            "AKI",
        ]:
            horizons = [int(h) for h in self.__getattribute__(clinical_task)]
            if not horizons:
                continue
            if clinical_task == "urine":
                # 데이터 부재로 -1 채움
                for h in horizons:
                    labeled_cohorts = labeled_cohorts.withColumn(
                        f"urine_{h}", F.lit(-1)
                    )
                continue

            labeled_cohorts = self.clinical_task(
                labeled_cohorts, clinical_task, horizons, spark
            )

        logger.info("Done preparing clinical task prediction for the given cohorts")
        labeled_cohorts = labeled_cohorts.toPandas()
        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")
        return labeled_cohorts

    def dialysis_filter(self, merge, timestamp_col: str):
        meta = self.task_itemids["dialysis"]
        proc = (
            merge.sql_ctx.read.csv(
                os.path.join(self.data_dir, meta["fname"]), header=True
            )
            if hasattr(merge, "sql_ctx")
            else None
        )
        if proc is None:
            return merge
        proc = proc.drop(*meta["exclude"])
        proc = proc.select(self.hadm_key, meta["timestamp"], meta["code"][0])
        proc = proc.filter(F.col(meta["code"][0]).isin(meta["itemid"]))
        proc = proc.withColumn("_proc_ts", F.to_timestamp(meta["timestamp"]))
        proc = proc.filter(F.col("_proc_ts").isNotNull())
        proc = (
            proc.groupBy(self.hadm_key)
            .agg(F.min("_proc_ts").alias("_proc_ts"))
            .withColumn("_dialysis_min", F.col("_proc_ts").cast("long") / 60)
        )
        merge = merge.join(proc, on=self.hadm_key, how="left")
        merge = merge.filter(
            F.col("_dialysis_min").isNull() | (F.col("_dialysis_min") > F.col(timestamp_col))
        )
        return merge.drop("_proc_ts", "_dialysis_min")

    def _normalize_units(self, table, task: str, value_col: str):
        unit_col = "valueuom"
        if unit_col not in table.columns:
            return table

        table = table.withColumn(unit_col, F.lower(F.col(unit_col)))

        if task == "creatinine":
            table = table.withColumn(
                value_col,
                F.when(F.col(unit_col).contains("umol"), F.col(value_col) / 88.42)
                .when(F.col(unit_col).contains("mg/l"), F.col(value_col) / 10.0)
                .otherwise(F.col(value_col)),
            )
        elif task in ["platelets", "wbc"]:
            raw_cells = (
                (F.col(unit_col).contains("cell"))
                | (F.col(unit_col).contains("/ul") & ~F.col(unit_col).rlike("10\\^|10e|10\\*|k/"))
            )
            table = table.withColumn(
                value_col,
                F.when(raw_cells, F.col(value_col) / 1000.0).otherwise(F.col(value_col)),
            )
        elif task == "hb":
            table = table.withColumn(
                value_col,
                F.when(F.col(unit_col).contains("mmol"), F.col(value_col) / 0.6206)
                .when(F.col(unit_col).contains("g/l"), F.col(value_col) / 10.0)
                .otherwise(F.col(value_col)),
            )
        elif task == "urine":
            table = table.withColumn(
                value_col,
                F.when(
                    F.col(unit_col).contains("l") & ~F.col(unit_col).contains("ml"),
                    F.col(value_col) * 1000.0,
                ).otherwise(F.col(value_col)),
            )

        return table

    def clinical_task(self, cohorts, task: str, horizons: List[int], spark):
        cfg = self.task_itemids[task]
        fname = cfg["fname"]
        timestamp = cfg["timestamp"]
        excludes = cfg["exclude"]
        code = cfg["code"][0]
        value = cfg["value"][0]
        itemid = cfg["itemid"]

        table = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
        table = table.drop(*excludes)
        table = table.withColumn(code, F.col(code).cast("int"))
        table = table.withColumn(value, F.col(value).cast("double"))
        table = table.withColumn(timestamp, F.to_timestamp(timestamp))
        table = table.filter(F.col(code).isin(itemid)).filter(F.col(value).isNotNull())
        table = table.filter(F.col(timestamp).isNotNull())
        table = self._normalize_units(table, task, value)

        merge = table.join(
            cohorts.select(self.hadm_key, self.icustay_key, "ADMITTIME", "INTIME", "weight"),
            on=self.hadm_key,
            how="inner",
        )
        merge = merge.withColumn("ADMITTIME_TS", F.to_timestamp("ADMITTIME"))
        merge = merge.withColumn(
            timestamp,
            (F.col(timestamp).cast("long") - F.col("ADMITTIME_TS").cast("long")) / 60,
        )

        if task in ["creatinine", "AKI"]:
            merge = self.dialysis_filter(merge, timestamp)

        merge = merge.filter(F.col(timestamp) >= F.col("INTIME") + self.pred_size * 60)

        baseline_lookup = None
        if task == "AKI":
            baseline_lookup = merge.filter(
                (F.col(timestamp) >= F.col("INTIME") - 7 * 24 * 60)
                & (F.col(timestamp) < F.col("INTIME") + 12 * 60)
            ).groupBy(self.icustay_key).agg(F.min(value).alias("baseline_creatinine"))

            icu_events = merge.filter(F.col(timestamp) >= F.col("INTIME"))
            window_first = Window.partitionBy(self.icustay_key).orderBy(timestamp)
            first_icu = (
                icu_events.withColumn("rn", F.row_number().over(window_first))
                .filter(F.col("rn") == 1)
                .select(self.icustay_key, F.col(value).alias("first_creatinine"))
            )
            baseline_lookup = baseline_lookup.join(
                F.broadcast(first_icu), on=self.icustay_key, how="outer"
            ).withColumn(
                "baseline_creatinine",
                F.coalesce(F.col("baseline_creatinine"), F.col("first_creatinine")),
            )

        for horizon in horizons:
            horizon_merge = merge.filter(
                F.col(timestamp)
                < F.col("INTIME") + (self.pred_size + horizon * 24) * 60
            ).filter(
                F.col(timestamp)
                >= F.col("INTIME") + (self.pred_size + (horizon - 1) * 24) * 60
            )

            horizon_agg = horizon_merge.groupby(self.icustay_key).agg(
                F.percentile_approx(F.col(value), 0.5).alias("value")
            )

            if task == "AKI" and baseline_lookup is not None:
                horizon_agg = horizon_agg.join(
                    F.broadcast(baseline_lookup.select(self.icustay_key, "baseline_creatinine")),
                    on=self.icustay_key,
                    how="left",
                )

            task_name = f"{task}_{horizon}"
            if task == "platelets":
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value >= 150, 0)
                    .when((horizon_agg.value >= 100) & (horizon_agg.value < 150), 1)
                    .when((horizon_agg.value >= 50) & (horizon_agg.value < 100), 2)
                    .when((horizon_agg.value >= 20) & (horizon_agg.value < 50), 3)
                    .when(horizon_agg.value < 20, 4),
                )
            elif task == "creatinine":
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value < 1.2, 0)
                    .when((horizon_agg.value >= 1.2) & (horizon_agg.value < 2.0), 1)
                    .when((horizon_agg.value >= 2.0) & (horizon_agg.value < 3.5), 2)
                    .when((horizon_agg.value >= 3.5) & (horizon_agg.value < 5), 3)
                    .when(horizon_agg.value >= 5, 4),
                )
            elif task == "wbc":
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value < 4, 0)
                    .when((horizon_agg.value >= 4) & (horizon_agg.value <= 12), 1)
                    .when((horizon_agg.value > 12), 2),
                )
            elif task == "hb":
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value < 8, 0)
                    .when((horizon_agg.value >= 8) & (horizon_agg.value < 10), 1)
                    .when((horizon_agg.value >= 10) & (horizon_agg.value < 12), 2)
                    .when((horizon_agg.value >= 12), 3),
                )
            elif task == "bicarbonate":
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when((horizon_agg.value < 22), 0)
                    .when((horizon_agg.value >= 22) & (horizon_agg.value < 29), 1)
                    .when((horizon_agg.value >= 29), 2),
                )
            elif task == "sodium":
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value < 135, 0)
                    .when((horizon_agg.value >= 135) & (horizon_agg.value < 145), 1)
                    .when((horizon_agg.value >= 145), 2),
                )
            elif task == "AKI":
                horizon_agg = horizon_agg.withColumn(
                    "rel_change",
                    (F.col("value") - F.col("baseline_creatinine"))
                    / F.col("baseline_creatinine"),
                )
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(F.col("rel_change") < 0.3, 0)
                    .when((F.col("rel_change") >= 0.3) & (F.col("rel_change") < 1.5), 1)
                    .when((F.col("rel_change") >= 1.5) & (F.col("rel_change") < 2.0), 2)
                    .when((F.col("rel_change") >= 2.0) & (F.col("rel_change") < 3.0), 3)
                    .when(F.col("rel_change") >= 3.0, 4),
                ).drop("rel_change")

            cohorts = cohorts.join(
                F.broadcast(horizon_agg.select(self.icustay_key, task_name)),
                on=self.icustay_key,
                how="left",
            )

        return cohorts
