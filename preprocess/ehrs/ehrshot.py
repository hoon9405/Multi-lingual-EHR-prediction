import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from ehrs import EHR, register_ehr

logger = logging.getLogger(__name__)


@register_ehr("ehrshot")
class EHRshot(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ehr_name = "ehrshot"

        if self.data_dir is None:
            raise ValueError(
                "EHRshot requires a local OMOP dump. Please provide --data pointing to the directory."
            )

        if self.ext is None:
            self.ext = ".csv"

        self._icustay_fname = "visit_detail" + self.ext
        self._patient_fname = "person" + self.ext
        self._admission_fname = "visit_occurrence" + self.ext

        self._icustay_key = "visit_detail_id"
        self._hadm_key = "visit_occurrence_id"
        self._patient_key = "person_id"
        self._determine_first_icu = "INTIME"

        # Concept sets
        self.creatinine_concepts = [3016723, 3017250, 3018311, 3001582]
        self.platelet_concepts = [3024929]
        self.wbc_concepts = [3006696, 3000905]
        self.hb_concepts = [3000963]
        self.bicarbonate_concepts = [3015632]
        self.sodium_concepts = [3019550]
        self.urine_concepts = [45876241]
        self.weight_concepts = [3025315, 3006322, 3013762]

        # Unit concept IDs
        self.unit_mg_dl = 8840
        self.unit_g_dl = 8713
        self.unit_k_per_ul = 8848
        self.unit_per_ul = 8647
        self.unit_k_per_ml = 9436
        self.unit_mmol_l = 8753
        self.unit_meq_l = 9557
        self.unit_ml = 8587
        self.unit_lb = 8739
        self.unit_oz = 9373
        self.unit_calc = 8596  # generic "calculated" unit some labs use for creatinine

        self.tables = [
            {
                "fname": "measurement" + self.ext,
                "timestamp": "measurement_DATETIME",
                "timeoffsetunit": "abs",
                "exclude": [
                    "Unnamed: 0",
                    "measurement_DATE",
                    "measurement_time",
                    "value_as_concept_id",
                    "range_low",
                    "range_high",
                    "provider_id",
                    "modifier_of_event_id",
                    "modifier_of_field_concept_id",
                    "trace_id",
                    "unit_id",
                    "load_table_id",
                ],
                "code": ["measurement_concept_id"],
                "value": ["value_as_number"],
            },
            {
                "fname": "observation" + self.ext,
                "timestamp": "observation_DATETIME",
                "timeoffsetunit": "abs",
                "exclude": [
                    "Unnamed: 0",
                    "observation_DATE",
                    "observation_type_concept_id",
                    "value_as_concept_id",
                    "qualifier_concept_id",
                    "provider_id",
                    "trace_id",
                    "unit_id",
                    "load_table_id",
                ],
                "code": ["observation_concept_id"],
                "value": ["value_as_number"],
            },
            {
                "fname": "procedure_occurrence" + self.ext,
                "timestamp": "procedure_DATETIME",
                "timeoffsetunit": "abs",
                "exclude": [
                    "Unnamed: 0",
                    "procedure_DATE",
                    "procedure_type_concept_id",
                    "modifier_concept_id",
                    "provider_id",
                    "trace_id",
                    "unit_id",
                    "load_table_id",
                ],
                "code": ["procedure_concept_id"],
            },
            {
                "fname": "drug_exposure" + self.ext,
                "timestamp": "drug_exposure_start_DATETIME",
                "timeoffsetunit": "abs",
                "exclude": [
                    "Unnamed: 0",
                    "drug_exposure_start_DATE",
                    "drug_exposure_end_DATE",
                    "drug_exposure_end_DATETIME",
                    "verbatim_end_DATE",
                    "trace_id",
                    "unit_id",
                    "load_table_id",
                ],
                "code": ["drug_concept_id"],
            },
        ]

        self.task_itemids = {
            "creatinine": {
                "fname": "measurement" + self.ext,
                "timestamp": "measurement_DATETIME",
                "exclude": ["Unnamed: 0"],
                "code": ["measurement_concept_id"],
                "value": ["value_as_number"],
                "itemid": self.creatinine_concepts,
            },
            "platelets": {
                "fname": "measurement" + self.ext,
                "timestamp": "measurement_DATETIME",
                "exclude": ["Unnamed: 0"],
                "code": ["measurement_concept_id"],
                "value": ["value_as_number"],
                "itemid": self.platelet_concepts,
            },
            "wbc": {
                "fname": "measurement" + self.ext,
                "timestamp": "measurement_DATETIME",
                "exclude": ["Unnamed: 0"],
                "code": ["measurement_concept_id"],
                "value": ["value_as_number"],
                "itemid": self.wbc_concepts,
            },
            "hb": {
                "fname": "measurement" + self.ext,
                "timestamp": "measurement_DATETIME",
                "exclude": ["Unnamed: 0"],
                "code": ["measurement_concept_id"],
                "value": ["value_as_number"],
                "itemid": self.hb_concepts,
            },
            "bicarbonate": {
                "fname": "measurement" + self.ext,
                "timestamp": "measurement_DATETIME",
                "exclude": ["Unnamed: 0"],
                "code": ["measurement_concept_id"],
                "value": ["value_as_number"],
                "itemid": self.bicarbonate_concepts,
            },
            "sodium": {
                "fname": "measurement" + self.ext,
                "timestamp": "measurement_DATETIME",
                "exclude": ["Unnamed: 0"],
                "code": ["measurement_concept_id"],
                "value": ["value_as_number"],
                "itemid": self.sodium_concepts,
            },
            "urine": {
                "fname": "measurement" + self.ext,
                "timestamp": "measurement_DATETIME",
                "exclude": ["Unnamed: 0"],
                "code": ["measurement_concept_id"],
                "value": ["value_as_number"],
                "itemid": self.urine_concepts,
            },
            "AKI": {
                "fname": "measurement" + self.ext,
                "timestamp": "measurement_DATETIME",
                "exclude": ["Unnamed: 0"],
                "code": ["measurement_concept_id"],
                "value": ["value_as_number"],
                "itemid": self.creatinine_concepts,
            },
            "dialysis": {
                "fname": "procedure_occurrence" + self.ext,
                "timestamp": "procedure_DATETIME",
                "exclude": ["Unnamed: 0"],
                "code": ["procedure_source_value"],
                "keywords": [
                    "dialysis",
                    "hemodialysis",
                    "hemofiltration",
                    "dialyze",
                    "crrt",
                    "continuous renal",
                    "peritoneal dialysis",
                ],
            },
        }

    def make_compatible(self, icustays: pd.DataFrame, spark):
        care_site = pd.read_csv(
            os.path.join(self.data_dir, "care_site" + self.ext),
            usecols=["care_site_id", "care_site_name"],
        )
        icustays = icustays.merge(care_site, on="care_site_id", how="left")

        icu_mask = (
            icustays["visit_detail_source_value"]
            .fillna("")
            .str.contains("ICU|Intensive Care|Critical Care", case=False, regex=True)
        ) | icustays["care_site_name"].fillna("").str.contains(
            "ICU|CCU|CICU|MICU|SICU|NICU|PICU|Critical Care", case=False, regex=True
        )
        icustays = icustays.loc[icu_mask].copy()

        visit_occurrence = pd.read_csv(
            os.path.join(self.data_dir, self._admission_fname),
            usecols=[
                "visit_occurrence_id",
                "visit_start_DATE",
                "visit_start_DATETIME",
                "visit_end_DATE",
                "visit_end_DATETIME",
            ],
        )
        visit_occurrence["visit_start"] = pd.to_datetime(
            visit_occurrence["visit_start_DATETIME"].fillna(
                visit_occurrence["visit_start_DATE"]
            ),
            errors="coerce",
        )

        icustays["icu_start"] = pd.to_datetime(
            icustays["visit_detail_start_DATETIME"].fillna(
                icustays["visit_detail_start_DATE"]
            ),
            errors="coerce",
        )
        icustays["icu_end"] = pd.to_datetime(
            icustays["visit_detail_end_DATETIME"].fillna(
                icustays["visit_detail_end_DATE"]
            ),
            errors="coerce",
        )

        icustays = icustays.merge(
            visit_occurrence[["visit_occurrence_id", "visit_start"]],
            on="visit_occurrence_id",
            how="left",
        )
        icustays.dropna(subset=["icu_start", "icu_end", "visit_start"], inplace=True)

        # Cast keys to numeric for consistent joins
        for col in [self.icustay_key, self._hadm_key, self._patient_key]:
            if col in icustays.columns:
                icustays[col] = pd.to_numeric(icustays[col], errors="coerce")

        icustays["ADMITTIME"] = icustays["visit_start"]
        icustays["INTIME"] = (
            icustays["icu_start"] - icustays["ADMITTIME"]
        ).dt.total_seconds() // 60
        icustays["LOS"] = (
            icustays["icu_end"] - icustays["icu_start"]
        ).dt.total_seconds() / 60 / 24

        person = pd.read_csv(
            os.path.join(self.data_dir, self._patient_fname),
            usecols=[
                "person_id",
                "gender_concept_id",
                "year_of_birth",
                "month_of_birth",
                "day_of_birth",
                "birth_DATETIME",
            ],
        )
        person["birth_ts"] = pd.to_datetime(person["birth_DATETIME"], errors="coerce")
        fallback_birth = pd.to_datetime(
            dict(
                year=person["year_of_birth"],
                month=person["month_of_birth"].fillna(1),
                day=person["day_of_birth"].fillna(1),
            ),
            errors="coerce",
        )
        person["birth_ts"] = person["birth_ts"].fillna(fallback_birth)
        icustays = icustays.merge(
            person[["person_id", "gender_concept_id", "birth_ts"]],
            on="person_id",
            how="left",
        )
        icustays["AGE"] = (
            (icustays["icu_start"] - icustays["birth_ts"]).dt.days / 365.25
        )

        death_path = os.path.join(self.data_dir, "death" + self.ext)
        if os.path.exists(death_path):
            death = pd.read_csv(
                death_path, usecols=["person_id", "death_DATE", "death_DATETIME"]
            )
            death["death_ts"] = pd.to_datetime(
                death["death_DATETIME"].fillna(death["death_DATE"]), errors="coerce"
            )
            icustays = icustays.merge(
                death[["person_id", "death_ts"]], on="person_id", how="left"
            )
            icustays["DEATHTIME"] = (
                icustays["death_ts"] - icustays["ADMITTIME"]
            ).dt.total_seconds() // 60
        else:
            icustays["DEATHTIME"] = np.nan

        # Weight from measurement table (converted to kg)
        weights = (
            spark.read.csv(
                os.path.join(self.data_dir, "measurement" + self.ext), header=True
            )
            .select(
                self.icustay_key,
                self.hadm_key,
                "measurement_concept_id",
                "value_as_number",
                "unit_concept_id",
            )
            .withColumn("measurement_concept_id", F.col("measurement_concept_id").cast("long"))
            .withColumn("unit_concept_id", F.col("unit_concept_id").cast("double"))
            .withColumn("value_as_number", F.col("value_as_number").cast("double"))
            .filter(F.col("measurement_concept_id").isin(self.weight_concepts))
        )
        weights = weights.withColumn(
            "weight_kg",
            F.when(F.col("unit_concept_id") == self.unit_lb, F.col("value_as_number") * 0.45359237)
            .when(F.col("unit_concept_id") == self.unit_oz, F.col("value_as_number") * 0.0283495231)
            .otherwise(F.col("value_as_number")),
        )
        weights = (
            weights.groupBy(self.icustay_key)
            .agg(F.avg("weight_kg").alias("weight"))
            .toPandas()
        )
        weights[self.icustay_key] = pd.to_numeric(weights[self.icustay_key], errors="coerce")
        icustays = icustays.merge(weights, on=self.icustay_key, how="left")
        icustays["weight"] = pd.to_numeric(icustays["weight"], errors="coerce")

        if "gender_concept_id" in icustays.columns:
            male_mask = (icustays["gender_concept_id"] == 8507) & icustays["weight"].isna()
            female_mask = (icustays["gender_concept_id"] == 8532) & icustays["weight"].isna()
            icustays.loc[male_mask, "weight"] = 80.0
            icustays.loc[female_mask, "weight"] = 65.0

        icustays.loc[icustays["weight"].isna(), "weight"] = 72.5
        icustays.loc[icustays["weight"] <= 0, "weight"] = 72.5

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
            if horizons:
                labeled_cohorts = self.clinical_task(
                    labeled_cohorts, clinical_task, horizons, spark
                )

        logger.info("Done preparing clinical task prediction for the given cohorts")
        labeled_cohorts = labeled_cohorts.toPandas()
        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")
        return labeled_cohorts

    def _normalize_units(self, df, task: str, value_col: str):
        df = df.withColumn("unit_concept_id", F.col("unit_concept_id").cast("double"))
        if task == "creatinine":
            df = df.withColumn(
                value_col,
                # Keep mg/dL and "calculated" entries; drop others to avoid mixing ratios
                F.when(
                    F.col("unit_concept_id").isin([self.unit_mg_dl, self.unit_calc]),
                    F.col(value_col),
                ),
            )
        elif task in ["platelets", "wbc"]:
            df = df.withColumn(
                value_col,
                F.when(F.col("unit_concept_id") == self.unit_k_per_ul, F.col(value_col))
                .when(
                    F.col("unit_concept_id").isin(
                        [self.unit_per_ul, self.unit_k_per_ml]
                    ),
                    F.col(value_col) / 1000.0,
                ),
            )
        elif task == "hb":
            df = df.withColumn(
                value_col,
                F.when(F.col("unit_concept_id") == self.unit_g_dl, F.col(value_col)),
            )
        elif task in ["bicarbonate", "sodium"]:
            df = df.withColumn(
                value_col,
                F.when(
                    F.col("unit_concept_id").isin(
                        [self.unit_mmol_l, self.unit_meq_l]
                    ),
                    F.col(value_col),
                ),
            )
        elif task == "urine":
            df = df.withColumn(
                value_col,
                F.when(F.col("unit_concept_id") == self.unit_ml, F.col(value_col)),
            )

        return df.filter(F.col(value_col).isNotNull())

    def dialysis_filter(self, merge, timestamp_col: str, spark):
        meta = self.task_itemids["dialysis"]
        proc = (
            spark.read.csv(os.path.join(self.data_dir, meta["fname"]), header=True)
            .drop(*meta["exclude"])
            .select(
                self.icustay_key,
                self.hadm_key,
                meta["timestamp"],
                meta["timestamp"].replace("DATETIME", "DATE"),
                meta["code"][0],
            )
        )
        pattern = "|".join(meta["keywords"])
        proc = proc.filter(F.col(meta["code"][0]).rlike(f"(?i){pattern}"))
        proc = proc.withColumn("_proc_ts", F.to_timestamp(meta["timestamp"]))
        proc = proc.withColumn(
            "_proc_ts",
            F.when(F.col("_proc_ts").isNull(), F.to_timestamp(meta["timestamp"].replace("DATETIME", "DATE"))).otherwise(
                F.col("_proc_ts")
            ),
        )
        proc = proc.filter(F.col("_proc_ts").isNotNull())

        proc = proc.join(
            merge.select(self.icustay_key, "ADMITTIME").dropDuplicates(
                [self.icustay_key]
            ),
            on=self.icustay_key,
            how="inner",
        )
        proc = proc.withColumn("ADMITTIME_TS", F.to_timestamp("ADMITTIME"))
        proc = proc.withColumn(
            "_DIALYSIS_TIME",
            F.round(
                (F.col("_proc_ts").cast("long") - F.col("ADMITTIME_TS").cast("long"))
                / 60
            ),
        )
        proc = (
            proc.groupBy(self.icustay_key)
            .agg(F.min("_DIALYSIS_TIME").alias("_DIALYSIS_TIME"))
        )
        merge = merge.join(F.broadcast(proc), on=self.icustay_key, how="left")
        merge = merge.filter(
            F.isnull("_DIALYSIS_TIME") | (F.col("_DIALYSIS_TIME") > F.col(timestamp_col))
        )
        return merge.drop("_DIALYSIS_TIME")

    def clinical_task(self, cohorts, task: str, horizons: List[int], spark):
        fname = self.task_itemids[task]["fname"]
        timestamp = self.task_itemids[task]["timestamp"]
        excludes = self.task_itemids[task]["exclude"]
        code = self.task_itemids[task]["code"][0]
        value = self.task_itemids[task]["value"][0] if "value" in self.task_itemids[task] else None
        itemid = self.task_itemids[task]["itemid"] if "itemid" in self.task_itemids[task] else None

        table = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
        table = table.drop(*excludes)

        if value:
            table = table.withColumn(value, F.col(value).cast("double"))
        table = table.withColumn(code, F.col(code).cast("long"))
        table = table.withColumn(timestamp, F.to_timestamp(timestamp))

        if itemid:
            table = table.filter(F.col(code).isin(itemid))
        if value:
            table = table.filter(F.col(value).isNotNull())
        table = table.filter(F.col(timestamp).isNotNull())

        if self.icustay_key in table.columns:
            table = table.drop(self.icustay_key)

        if self.hadm_key in table.columns:
            table = table.withColumn(self.hadm_key, F.col(self.hadm_key).cast("long"))

        merge = table.join(
            cohorts.select(self.icustay_key, self.hadm_key, "ADMITTIME", "INTIME", "weight"),
            on=self.hadm_key,
            how="inner",
        )
        merge = merge.withColumn("ADMITTIME_TS", F.to_timestamp("ADMITTIME"))
        merge = merge.withColumn(
            timestamp,
            F.round(
                (F.col(timestamp).cast("long") - F.col("ADMITTIME_TS").cast("long"))
                / 60
            ),
        )

        if value:
            merge = self._normalize_units(merge, task, value)
            merge = merge.filter(F.col(value).isNotNull())

        if task in ["creatinine", "urine", "AKI"]:
            merge = self.dialysis_filter(merge, timestamp, spark)

        baseline_lookup = None
        if task == "AKI":
            baseline_lookup = merge.filter(
                (F.col(timestamp) >= F.col("INTIME") - 7 * 24 * 60)
                & (F.col(timestamp) < F.col("INTIME") + 12 * 60)
            ).groupBy(self.icustay_key).agg(
                F.min(value).alias("baseline_creatinine")
            )

            icu_events = merge.filter(F.col(timestamp) >= F.col("INTIME"))
            window_first = Window.partitionBy(self.icustay_key).orderBy(timestamp)
            first_icu = (
                icu_events.withColumn("rn", F.row_number().over(window_first))
                .filter(F.col("rn") == 1)
                .select(self.icustay_key, F.col(value).alias("first_creatinine"))
            )
            baseline_lookup = baseline_lookup.join(
                F.broadcast(first_icu),
                on=self.icustay_key,
                how="outer",
            ).withColumn(
                "baseline_creatinine",
                F.coalesce(F.col("baseline_creatinine"), F.col("first_creatinine")),
            ).select(self.icustay_key, "baseline_creatinine")

        merge = merge.filter(F.col(timestamp) >= F.col("INTIME") + self.pred_size * 60)

        weight_lookup = None
        if "weight" in merge.columns:
            weight_lookup = merge.select(
                self.icustay_key, "weight"
            ).dropDuplicates([self.icustay_key])

        for horizon in horizons:
            horizon_merge = merge.filter(
                F.col(timestamp)
                < F.col("INTIME") + (self.pred_size + horizon * 24) * 60
            ).filter(
                F.col(timestamp)
                >= F.col("INTIME") + (self.pred_size + (horizon - 1) * 24) * 60
            )
            agg_column = value
            if task == "urine" and weight_lookup is not None:
                horizon_merge = horizon_merge.withColumn("_URINE_RAW", F.col(value))
                agg_column = "_URINE_RAW"

            horizon_agg = horizon_merge.groupby(self.icustay_key).agg(
                F.percentile_approx(F.col(agg_column), 0.5).alias("value")
            )

            if task == "urine" and weight_lookup is not None:
                horizon_agg = horizon_agg.join(
                    F.broadcast(weight_lookup), on=self.icustay_key, how="left"
                )
                horizon_agg = horizon_agg.withColumn(
                    "value",
                    F.when(F.col("weight").isNotNull(), F.col("value") / F.col("weight"))
                    .otherwise(F.col("value")),
                ).drop("weight")

            if task == "AKI" and baseline_lookup is not None:
                horizon_agg = horizon_agg.join(
                    F.broadcast(baseline_lookup), on=self.icustay_key, how="left"
                )
                horizon_agg = horizon_agg.filter(
                    F.col("baseline_creatinine").isNotNull()
                    & (F.col("baseline_creatinine") > 0)
                )

            task_name = task + "_" + str(horizon)
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

            elif task == "urine":
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value >= 0.5, 0)
                    .when((horizon_agg.value >= 0.3) & (horizon_agg.value < 0.5), 1)
                    .when((horizon_agg.value >= 0.1) & (horizon_agg.value < 0.3), 2)
                    .when((horizon_agg.value >= 0.05) & (horizon_agg.value < 0.1), 3)
                    .when(horizon_agg.value < 0.05, 4),
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
            if task == "AKI" and f"urine_{horizon}" in cohorts.columns:
                cohorts = cohorts.withColumn(
                    task_name, F.greatest(F.col(task_name), F.col(f"urine_{horizon}"))
                )

        return cohorts
