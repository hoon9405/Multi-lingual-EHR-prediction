import glob
import logging
import os
from functools import reduce

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window

from ehrs import EHR, register_ehr

logger = logging.getLogger(__name__)


@register_ehr("umcdb")
class UMCdb(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ehr_name = "umcdb"

        if self.data_dir is None:
            raise ValueError(
                "Cache is not supported for UMCdb. Please provide the data directory."
            )

        logger.info("Data directory is set to {}".format(self.data_dir))

        if self.ext is None:
            self.ext = self.infer_data_extension()

        self._icustay_fname = "admissions" + self.ext
        self._diagnosis_fname = "listitems" + self.ext

        self.tables = [
            {
                "fname": "drugitems" + self.ext,
                "timestamp": "start",
                "timeoffsetunit": "ms",
                "exclude": [
                    "orderid",
                    "ordercategoryid",
                    "itemid",
                    "rateunitid",
                    "ratetimeunitid",
                    "doseunitid",
                    "doserateunitid",
                    "administeredunitid",
                    "stop",
                    "solutionitemid",
                ],
            },
            {
                "fname": "freetextitems" + self.ext,
                "timestamp": "measuredat",
                "timeoffsetunit": "ms",
                "exclude": [
                    "itemid",
                    "registeredat",
                    "registeredby",
                    "updatedat",
                    "updateby",
                ],
            },
            {
                "fname": "listitems" + self.ext,
                "timestamp": "measuredat",
                "timeoffsetunit": "ms",
                "exclude": [
                    "itemid",
                    "registeredat",
                    "updatedat",
                    "updatedby",
                ],
            },
            {
                "fname": "numericitems" + self.ext,
                "timestamp": "measuredat",
                "timeoffsetunit": "ms",
                "exclude": [
                    "itemid",
                    "unitid",
                    "registeredat",
                    "updatedat",
                    "updatedby",
                ],
            },
            {
                "fname": "procedureorderitems" + self.ext,
                "timestamp": "registeredat",
                "timeoffsetunit": "ms",
                "exclude": ["orderid", "ordercategoryid", "itemid"],
            },
            {
                "fname": "processitems" + self.ext,
                "timestamp": "start",
                "timeoffsetunit": "ms",
                "exclude": ["itemid", "stop"],
            },
        ]

        if (
            self.creatinine
            or self.platelets
            or self.wbc
            or self.hb
            or self.bicarbonate
            or self.sodium
            or self.urine
            or self.AKI
        ):
            self.task_itemids = {
                "creatinine": {
                    "fname": "numericitems" + self.ext,
                    "timestamp": "measuredat",
                    "timeoffsetunit": "ms",
                    "code": ["itemid"],
                    "value": ["value"],
                    # https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/master/amsterdamumcdb/sql/common/creatinine_acute_kidney_injury_failure.sql
                    "itemid": [6836, 9941, 14216],
                },
                "platelets": {
                    "fname": "numericitems" + self.ext,
                    "timestamp": "measuredat",
                    "timeoffsetunit": "ms",
                    "code": ["itemid"],
                    "value": ["value"],
                    # https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/master/amsterdamumcdb/sql/common/platelets.sql
                    "itemid": [9964, 6797, 10409, 14252],
                },
                "wbc": {
                    "fname": "numericitems" + self.ext,
                    "timestamp": "measuredat",
                    "timeoffsetunit": "ms",
                    "code": ["itemid"],
                    "value": ["value"],
                    "itemid": [6779, 9965],
                },
                "dialysis": {
                    "tables": {
                        "numericitems": {
                            "fname": "numericitems" + self.ext,
                            "timestamp": "measuredat",
                            "timeoffsetunit": "ms",
                            "itemid": [
                                7667,
                                7668,
                                7671,
                                7905,
                                8622,
                                8805,
                                10735,
                                10736,
                                12091,
                                12301,
                                12444,
                                12445,
                                12446,
                                12447,
                                12448,
                                12449,
                                12450,
                                12451,
                                12452,
                                12453,
                                12454,
                                12455,
                                12456,
                                12458,
                                12459,
                                12460,
                                12461,
                                12463,
                                8806,
                                14835,
                                14836,
                                14837,
                                14838,
                                14839,
                                14840,
                                14841,
                                14842,
                                14843,
                                14844,
                                14845,
                                14848,
                                14849,
                                14850,
                                14851,
                                14852,
                                20076,
                                20077,
                                20078,
                                20079,
                                20080,
                                20340,
                                20537,
                                20538,
                                20539,
                                20543,
                                20544,
                                20547,
                                20706,
                                20707,
                                20708,
                                20709,
                                20710,
                                20716,
                            ],
                        },
                        "listitems": {
                            "fname": "listitems" + self.ext,
                            "timestamp": "measuredat",
                            "timeoffsetunit": "ms",
                            "itemid": [
                                7672,
                                7969,
                                8616,
                                10659,
                                20536,
                                20542,
                                20546,
                                20548,
                                20549,
                            ],
                        },
                        "processitems": {
                            "fname": "processitems" + self.ext,
                            "timestamp": "start",
                            "timeoffsetunit": "ms",
                            "itemid": [12465, 9162, 9161, 16363, 9163],
                        },
                    }
                },
                "hb": {
                    "fname": "numericitems" + self.ext,
                    "timestamp": "measuredat",
                    "timeoffsetunit": "ms",
                    "code": ["itemid"],
                    "value": ["value"],
                    "itemid": [9960, 6778, 10286, 19703, 9553],
                },
                "bicarbonate": {
                    "fname": "numericitems" + self.ext,
                    "timestamp": "measuredat",
                    "timeoffsetunit": "ms",
                    "code": ["itemid"],
                    "value": ["value"],
                    "itemid": [9992, 6810],
                },
                "sodium": {
                    "fname": "numericitems" + self.ext,
                    "timestamp": "measuredat",
                    "timeoffsetunit": "ms",
                    "code": ["itemid"],
                    "value": ["value"],
                    "itemid": [9924, 6840, 9555, 10284],
                },
                "urine": {
                    "fname": "numericitems" + self.ext,
                    "timestamp": "measuredat",
                    "timeoffsetunit": "ms",
                    "code": ["itemid"],
                    "value": ["value"],
                    "itemid": [8794, 8796, 8798, 8800, 8803],
                },
                "AKI": {
                    "fname": "numericitems" + self.ext,
                    "timestamp": "measuredat",
                    "timeoffsetunit": "ms",
                    "code": ["itemid"],
                    "value": ["value"],
                    "itemid": [6836, 9941, 14216],
                },
            }

        # if cfg.use_ed:
        #     raise NotImplementedError()

        self._icustay_key = "admissionid"
        self._hadm_key = None
        self._patient_key = "patientid"

        self._determine_first_icu = "admissioncount"

    def prepare_tasks(self, cohorts, spark, cached=False):
        labeled_cohorts = super().prepare_tasks(cohorts, spark, cached)
        if cached:
            return labeled_cohorts
        
        if self.diagnosis:
            logger.info("Start labeling cohorts for diagnosis prediction.")

            diagnoses = pd.read_csv(
                os.path.join(self.data_dir, self.diagnosis_fname), encoding="ISO-8859-1"
            )
            diagnoses = diagnoses[
                diagnoses["itemid"].isin([13110, 16651, 18588, 16997, 18669, 18671])
            ]

            def get_diagnosis_category(x):
                value = x["value"]
                if x["itemid"] in [18669, 18671]:
                    value = value.split("-")[0]

                value = value.lower()
                value = value.split("operative")[-1]
                value = value.split("operatief")[-1]
                value = value.strip()

                categories = {
                    "cardiovascular": 0,  # Cardiovascular
                    "cardiovasculair": 0,  # Cardiovascular
                    "thoraxchirurgie": 1,  # General Surgery
                    "respiratory": 2,  # Respiratory
                    "respiratoir": 2,  # Respiratory
                    "neurochirurgie": 3,  # Neurological
                    "neurologic": 3,  # Neurological
                    "neurologisch": 3,  # Neurological
                    "neurologie": 3,  # Neurological
                    "algemene chirurgie": 1,  # General Surgery
                    "genitourinary": 4,  # Genitourinary/Renal
                    "renaal": 4,  # Genitourinary/Renal
                    "gastro-intestinal": 5,  # Gastrointestinal
                    "gastro-intestinaal": 5,  # Gastrointestinal
                    "gastro": 5,  # Gastrointestinal
                    "hematological": 6,  # Hematological
                    "hematologisch": 6,  # Hematological
                    "hematology": 6,  # Hematological
                    "transplant": 7,  # Transplant
                    "trauma": 8,  # Trauma
                    "metabolic": 9,  # Metabolic
                    "metabolisme": 9,  # Metabolic
                    "musculoskeletal /skin": 10,  # Musculoskeletal/Skin
                    "musculo-skeletal": 10,  # Musculoskeletal/Skin
                    "interne geneeskunde": 11,  # Internal Medicine
                    "interne geneeskunde _old_1": 11,  # Internal Medicine
                    "post": 12,  # Non-Categorized/General"
                    "non": 12,  # Non-Categorized/General"
                }
                value = categories[value]

                return value

            diagnoses["diagnosis"] = diagnoses.apply(get_diagnosis_category, axis=1)
            diagnoses = (
                diagnoses.groupby(self.icustay_key)["diagnosis"]
                .agg(lambda x: list(set(x)))
                .to_frame()
            )

            labeled_cohorts = labeled_cohorts.merge(
                diagnoses, on=self.icustay_key, how="left"  # Use all
            )
            labeled_cohorts["diagnosis"] = labeled_cohorts["diagnosis"].map(
                lambda x: x if isinstance(x, list) else []
            )

            logger.info("Done preparing diagnosis prediction for the given cohorts")

            self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")

        logger.info("Start labeling cohorts for clinical task prediction.")
        base_cohorts_pdf = labeled_cohorts
        labeled_cohorts_sdf = spark.createDataFrame(base_cohorts_pdf)

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

        for clinical_task in clinical_tasks:
            horizons = self.__getattribute__(clinical_task)
            if not horizons:
                continue

            logger.info(f"Start clinical task labeling for {clinical_task} (horizons={horizons})")
            task_labeled = self.clinical_task(
                labeled_cohorts_sdf,
                clinical_task,
                horizons,
                spark,
            )

            task_cols = [self.icustay_key] + [
                c for c in task_labeled.columns if c.startswith(f"{clinical_task}_")
            ]
            if len(task_cols) <= 1:
                continue

            task_pdf = task_labeled.select(*task_cols).toPandas()
            base_cohorts_pdf = base_cohorts_pdf.merge(task_pdf, on=self.icustay_key, how="left")

        logger.info("Done preparing clinical task prediction for the given cohorts")
        labeled_cohorts = base_cohorts_pdf
        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")
        return labeled_cohorts

    def make_compatible(self, icustays, spark):
  
        # prepare icustays according to the appropriate format
        icustays["LOS"] = icustays["lengthofstay"].map(lambda x: x / 24)
        icustays["ADMITTIME"] = 0
        icustays["INTIME"] = icustays["admittedat"] / 1000 / 60
        icustays["AGE"] = 100  # Only adults in dataset, Just for compatibility

        icustays["OUTTIME"] = (
            icustays["dischargedat"] / 1000 / 60 - icustays["INTIME"]
        )  # Milliseconds to minutes
        icustays["DEATHTIME"] = icustays.apply(
            lambda x: (
                x["dischargedat"] / 1000 / 60
                if x["destination"] == "Overleden"
                else np.nan
            ),
            axis=1,
        )
        icustays["readmission"] = None

        weight_col = None
        for cand in [
            "weight",
            "Weight",
            "admissionweight",
            "admissionWeight",
            "weightvalue",
            "actualweight",
        ]:
            if cand in icustays.columns:
                weight_col = cand
                break

        if weight_col is None:
            icustays["weight"] = np.nan
        elif weight_col != "weight":
            icustays = icustays.rename(columns={weight_col: "weight"})

        icustays["weight"] = pd.to_numeric(icustays.get("weight", np.nan), errors="coerce")

        gender_col = None
        for cand in ["gender", "Gender", "sex", "Sex"]:
            if cand in icustays.columns:
                gender_col = cand
                break

        if gender_col is not None:
            gender_series = icustays[gender_col].astype(str).str.upper()
            male_mask = icustays["weight"].isna() & gender_series.str.startswith("M")
            female_mask = icustays["weight"].isna() & gender_series.str.startswith("F")
            icustays.loc[male_mask, "weight"] = 80.0
            icustays.loc[female_mask, "weight"] = 65.0

        icustays.loc[icustays["weight"].isna(), "weight"] = 72.5
        icustays.loc[icustays["weight"] <= 0, "weight"] = 72.5

        return icustays

    def clinical_task(self, cohorts, task, horizons, spark):
        fname = self.task_itemids[task]["fname"]
        timestamp = self.task_itemids[task]["timestamp"]
        code = self.task_itemids[task]["code"][0]
        value = self.task_itemids[task]["value"][0]
        itemid = self.task_itemids[task]["itemid"]

        table = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
        table = table.select(self.icustay_key, code, value, timestamp)
        table = table.withColumn(timestamp, F.col(timestamp).cast("double"))
        table = table.withColumn(value, F.col(value).cast("double"))
        table.cache()
        table = table.filter(F.col(code).isin(itemid)).filter(F.col(value).isNotNull())

        cohort_cols = ["ADMITTIME", "INTIME"]
        if "weight" in cohorts.columns:
            cohort_cols.append("weight")

        merge = table.join(
            F.broadcast(cohorts.select(self.icustay_key, *cohort_cols)),
            on=self.icustay_key,
            how="inner",
        )

        if task in ["creatinine", "urine", "AKI"]:
            dialysis_tables = []
            for dialysis_dict in self.task_itemids["dialysis"]["tables"].values():
                dialysis_table = spark.read.csv(
                    os.path.join(self.data_dir, dialysis_dict["fname"]), header=True
                )
                dialysis_table = dialysis_table.select(
                    self.icustay_key, "itemid", dialysis_dict["timestamp"]
                )
                dialysis_table = dialysis_table.filter(
                    F.col("itemid").isin(dialysis_dict["itemid"])
                )
                dialysis_table = dialysis_table.withColumn(
                    "_DIALYSIS_TIME",
                    F.col(dialysis_dict["timestamp"]) / 1000 / 60,
                )
                dialysis_tables.append(
                    dialysis_table.select(self.icustay_key, "_DIALYSIS_TIME")
                )
            dialysis = reduce(lambda x, y: x.union(y), dialysis_tables)

            dialysis = dialysis.groupby(self.icustay_key).agg(
                F.min("_DIALYSIS_TIME").alias("_DIALYSIS_TIME")
            )
            merge = merge.join(dialysis, on=self.icustay_key, how="left")
            # Only leave events with no dialysis / before first dialysis
            merge = merge.filter(
                F.isnull("_DIALYSIS_TIME")
                | (F.col("_DIALYSIS_TIME") > F.col(timestamp))
            )
            merge = merge.drop("_DIALYSIS_TIME")

        merge = merge.withColumn(
            timestamp, F.round((F.col(timestamp) / 1000 / 60 - F.col("ADMITTIME")))
        )

        if "weight" in merge.columns:
            merge = merge.withColumn("weight", F.col("weight").cast("double"))

        baseline_lookup = None
        if task == "AKI":
            baseline_lookup = merge.filter(
                (F.col(timestamp) >= F.col("INTIME") - 7 * 24 * 60)
                & (F.col(timestamp) < F.col("INTIME") + 12 * 60)
            ).groupBy(self.icustay_key).agg(
                F.min("value").alias("baseline_creatinine")
            )

            # Fallback: ICU 입실 이후 첫 creatinine 측정을 baseline으로 사용
            icu_events = merge.filter(F.col(timestamp) >= F.col("INTIME"))
            window_first = Window.partitionBy(self.icustay_key).orderBy(timestamp)
            first_icu = (
                icu_events.withColumn("rn", F.row_number().over(window_first))
                .filter(F.col("rn") == 1)
                .select(self.icustay_key, F.col("value").alias("first_creatinine"))
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
            weight_lookup = merge.select(self.icustay_key, "weight").dropDuplicates([self.icustay_key])

        for horizon in horizons:
            horizon_merge = merge.filter(
                (F.col(timestamp)
                 < F.col("INTIME") + (self.pred_size + horizon * 24) * 60)
            ).filter(
                F.col(timestamp)
                >= F.col("INTIME") + (self.pred_size + (horizon - 1) * 24) * 60
            )

            task_name = task + "_" + str(horizon)

            horizon_agg = horizon_merge.groupby(self.icustay_key).agg(
                F.percentile_approx(F.col(value), 0.5).alias("value")
            )

            if task == "urine" and weight_lookup is not None:
                horizon_agg = horizon_agg.join(
                    F.broadcast(weight_lookup),
                    on=self.icustay_key,
                    how="left",
                )
                horizon_agg = horizon_agg.withColumn(
                    "value",
                    F.when(F.col("weight").isNotNull(), F.col("value") / F.col("weight"))
                    .otherwise(F.col("value"))
                ).drop("weight")

            if task == "AKI" and baseline_lookup is not None:
                horizon_agg = horizon_agg.join(
                    F.broadcast(baseline_lookup),
                    on=self.icustay_key,
                    how="left",
                )

            # Labeling
            if task == "platelets":
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value >= 150, 0)
                    .when((horizon_agg.value >= 100) & (horizon_agg.value < 150), 1)
                    .when((horizon_agg.value >= 50) & (horizon_agg.value < 100), 2)
                    .when((horizon_agg.value >= 20) & (horizon_agg.value < 50), 3)
                    .when(horizon_agg.value < 20, 4),
                )

            elif task == "creatinine":  # mg/dL -> umol/L
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value < 1.2 * 88.42, 0)
                    .when(
                        (horizon_agg.value >= 1.2 * 88.42)
                        & (horizon_agg.value < 2.0 * 88.42),
                        1,
                    )
                    .when(
                        (horizon_agg.value >= 2.0 * 88.42)
                        & (horizon_agg.value < 3.5 * 88.42),
                        2,
                    )
                    .when(
                        (horizon_agg.value >= 3.5 * 88.42)
                        & (horizon_agg.value < 5 * 88.42),
                        3,
                    )
                    .when(horizon_agg.value >= 5 * 88.42, 4),
                )

            elif task == "wbc":
                # NOTE: unit is mg/L
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value < 4, 0)
                    .when((horizon_agg.value >= 4) & (horizon_agg.value <= 12), 1)
                    .when((horizon_agg.value > 12), 2),
                )

            elif task == "hb":  # g/dL to mmol/L
                horizon_agg = horizon_agg.withColumn(
                    task_name,
                    F.when(horizon_agg.value < 8 * 0.6206, 0)
                    .when(
                        (horizon_agg.value >= 8 * 0.6206)
                        & (horizon_agg.value < 10 * 0.6206),
                        1,
                    )
                    .when(
                        (horizon_agg.value >= 10 * 0.6206)
                        & (horizon_agg.value < 12 * 0.6206),
                        2,
                    )
                    .when((horizon_agg.value >= 12 * 0.6206), 3),
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
                    .when(F.col("rel_change") >= 3.0, 4)
                ).drop("rel_change")

            cohorts = cohorts.join(
                F.broadcast(horizon_agg.select(self.icustay_key, task_name)),
                on=self.icustay_key,
                how="left",
            )
            if task == "AKI" and f"urine_{horizon}" in cohorts.columns:
                cohorts = cohorts.withColumn(
                    task_name,
                    F.greatest(F.col(task_name), F.col(f"urine_{horizon}")),
                )

        return cohorts

    def infer_data_extension(self) -> str:
        if len(glob.glob(os.path.join(self.data_dir, "*.csv.gz"))) == 7:
            ext = ".csv.gz"
        elif len(glob.glob(os.path.join(self.data_dir, "*.csv"))) == 7:
            ext = ".csv"
        else:
            raise AssertionError(
                "Provided data directory is not correct. Please check if --data is correct. "
                "--data: {}".format(self.data_dir)
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext
