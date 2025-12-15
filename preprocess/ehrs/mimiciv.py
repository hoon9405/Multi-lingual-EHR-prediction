import glob
import logging
import os
import csv
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window
import random
from ehrs import EHR, register_ehr
import time

logger = logging.getLogger(__name__)

@register_ehr("mimiciv")
class MIMICIV(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ehr_name = "mimiciv"

        if self.data_dir is None:
            self.data_dir = os.path.join(self.cache_dir, self.ehr_name)

            if not os.path.exists(self.data_dir):
                logger.info(
                    "Data is not found so try to download from the internet. "
                    "Note that this is a restricted-access resource. "
                    "Please log in to physionet.org with a credentialed user."
                )
                self.download_ehr_from_url(
                    url="https://physionet.org/files/mimiciv/2.0/", dest=self.data_dir
                )

        logger.info("Data directory is set to {}".format(self.data_dir))

        if self.ccs_path is None:
            self.ccs_path = os.path.join(self.cache_dir, "ccs_multi_dx_tool_2015.csv")

            if not os.path.exists(self.ccs_path):
                logger.info(
                    "`ccs_multi_dx_tool_2015.csv` is not found so try to download from the internet."
                )
                self.download_ccs_from_url(self.cache_dir)

        if self.gem_path is None:
            self.gem_path = os.path.join(self.cache_dir, "icd10cmtoicd9gem.csv")

            if not os.path.exists(self.gem_path):
                logger.info(
                    "`icd10cmtoicd9gem.csv` is not found so try to download from the internet."
                )
                self.download_icdgem_from_url(self.cache_dir)

        if self.ext is None:
            self.ext = self.infer_data_extension()

        self._icustay_fname = "icu/icustays" + self.ext
        self._patient_fname = "hosp/patients" + self.ext
        self._admission_fname = "hosp/admissions" + self.ext
        self._diagnosis_fname = "hosp/diagnoses_icd" + self.ext

        self.tables = [
            {
                "fname": "hosp/labevents" + self.ext,
                "timestamp": "charttime",
                "timeoffsetunit": "abs",
                "exclude": [
                    "labevent_id",
                    "storetime",
                    "subject_id",
                    "specimen_id",
                    "order_provider_id"
                    ],
                "code": ["itemid"],
                "desc": ["hosp/d_labitems" + self.ext],
                "desc_key": [["label"]],
                "desc_code_col": ["itemid"],
                "rename_map": [{"label": "itemid"}],
            },
            {
                "fname": "hosp/prescriptions" + self.ext,
                "timestamp": "starttime",
                "timeoffsetunit": "abs",
                "exclude": [
                    "gsn",
                    "ndc",
                    "subject_id",
                    "pharmacy_id",
                    "poe_id",
                    "poe_seq",
                    "formulary_drug_cd",
                    "stoptime",
                    "order_provider_id"
                ],
            },
            {
                "fname": "icu/inputevents" + self.ext,
                "timestamp": "starttime",
                "timeoffsetunit": "abs",
                "exclude": [
                    "endtime",
                    "storetime",
                    "orderid",
                    "linkorderid",
                    "subject_id",
                    "continueinnextdept",
                    "statusdescription",
                ],
                "code": ["itemid"],
                "desc": ["icu/d_items" + self.ext],
                "desc_key": [["label"]],
                "desc_code_col": ["itemid"],
                "rename_map": [{"label": "itemid"}],
            },
        ]

        if cfg.use_more_tables:
            self.tables += [
                {
                    "fname": "icu/chartevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "storetime",
                        "subject_id",
                    ],
                    "code": ["itemid"],
                    "desc": ["icu/d_items" + self.ext],
                    "desc_key": [["label"]],
                    "desc_code_col": ["itemid"],
                    "rename_map": [{"label": "itemid"}],
                },
                {
                    "fname": "icu/outputevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "storetime",
                        "subject_id",
                    ],
                    "code": ["itemid"],
                    "desc": ["icu/d_items" + self.ext],
                    "desc_key": [["label"]],
                    "desc_code_col": ["itemid"],
                    "rename_map": [{"label": "itemid"}],
                },
                {
                    "fname": "hosp/microbiologyevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "chartdate",
                        "storetime",
                        "storedate",
                        "subject_id",
                        "microevent_id",
                        "micro_specimen_id",
                        "spec_itemid",
                        "test_itemid",
                        "org_itemid",
                        "ab_itemid",
                    ],
                },
                {
                    "fname": "icu/procedureevents" + self.ext,
                    "timestamp": "starttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "storetime",
                        "endtime",
                        "subject_id",
                        "orderid",
                        "linkorderid",
                        "continueinnextdept",
                        "statusdescription",
                    ],
                    "code": ["itemid"],
                    "desc": ["icu/d_items" + self.ext],
                    "desc_key": [["label"]],
                    "desc_code_col": ["itemid"],
                    "rename_map": [{"label": "itemid"}],
                },
            ]

        if cfg.use_ed:
            self._ed_fname = "ed/edstays" + self.ext
            self._ed_key = "stay_id"
            self.tables += [
                {
                    "fname": "ed/medrecon" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": ["gsn", "ndc", "etc_rn", "etccode", "subject_id"],
                },
                {
                    "fname": "ed/pyxis" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": ["med_rn", "gsn", "gsn_rn", "subject_id"],
                },
                {
                    "fname": "ed/vitalsign" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": ["subject_id"],
                },
                {
                    "fname": "ed/diagnosis" + self.ext,
                    "timestamp": "ED_OUTTIME",
                    "timeoffsetunit": "abs",
                    "exclude": ["subject_id", "icd_code"],
                },
                {
                    "fname": "ed/triage" + self.ext,
                    "timestamp": "ED_INTIME",
                    "timeoffsetunit": "abs",
                    "exclude": ["subject_id"],
                },
            ]
            
        # clinical test define
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
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "labevent_id",
                        "subject_id",
                        "specimen_id",
                        "storetime",
                        "value",
                        "valueuom",
                        "ref_range_lower",
                        "ref_range_upper",
                        "flag",
                        "priority",
                        "comments",
                    ],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [50912],
                },
                "platelets": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "labevent_id",
                        "subject_id",
                        "specimen_id",
                        "storetime",
                        "value",
                        "valueuom",
                        "ref_range_lower",
                        "ref_range_upper",
                        "flag",
                        "priority",
                        "comments",
                    ],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [51265],
                },
                "wbc": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "labevent_id",
                        "subject_id",
                        "specimen_id",
                        "storetime",
                        "value",
                        "valueuom",
                        "ref_range_lower",
                        "ref_range_upper",
                        "flag",
                        "priority",
                        "comments",
                    ],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [51300, 51301, 51755],
                },
                "dialysis": {
                    "tables": {
                        "chartevents": {
                            "fname": "icu/chartevents" + self.ext,
                            "timestamp": "charttime",
                            "timeoffsetunit": "abs",
                            "include": ["subject_id", "itemid", "value", "charttime"],
                            "itemid": {
                                "ce": [
                                    226499,
                                    224154,
                                    225183,
                                    227438,
                                    224191,
                                    225806,
                                    225807,
                                    228004,
                                    228005,
                                    228006,
                                    224144,
                                    224145,
                                    224153,
                                    226457,
                                ]
                            },
                        },
                        "inputevents": {
                            "fname": "icu/inputevents" + self.ext,
                            "timestamp": "starttime",
                            "timeoffsetunit": "abs",
                            "include": ["subject_id", "itemid", "amount", "starttime"],
                            "itemid": {"ie": [227536, 227525]},
                        },
                        "procedureevents": {
                            "fname": "icu/procedureevents" + self.ext,
                            "timestamp": "starttime",
                            "timeoffsetunit": "abs",
                            "include": ["subject_id", "itemid", "value", "starttime"],
                            "itemid": {
                                "pe": [225441, 225802, 225803, 225805, 225809, 225955]
                            },
                        },
                    }
                },
                "hb": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "labevent_id",
                        "subject_id",
                        "specimen_id",
                        "storetime",
                        "value",
                        "valueuom",
                        "ref_range_lower",
                        "ref_range_upper",
                        "flag",
                        "priority",
                        "comments",
                    ],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [51222],
                },
                "bicarbonate": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "labevent_id",
                        "subject_id",
                        "specimen_id",
                        "storetime",
                        "value",
                        "valueuom",
                        "ref_range_lower",
                        "ref_range_upper",
                        "flag",
                        "priority",
                        "comments",
                    ],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [50882],
                },
                "sodium": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "labevent_id",
                        "subject_id",
                        "specimen_id",
                        "storetime",
                        "value",
                        "valueuom",
                        "ref_range_lower",
                        "ref_range_upper",
                        "flag",
                        "priority",
                        "comments",
                    ],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [50983],
                },
                "urine": {
                    "fname": "icu/outputevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "subject_id",
                        "valueuom",
                        "storetime",
                    ],
                    "code": ["itemid"],
                    "value": ["value"],
                    "itemid": [
                        226557, 226558, 226559, 226560, 226561, 
                        226563, 226564, 226565, 226566, 226567, 
                        226584, 227510
                    ],
                },
                "AKI": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "labevent_id",
                        "subject_id",
                        "specimen_id",
                        "storetime",
                        "value",
                        "valueuom",
                        "ref_range_lower",
                        "ref_range_upper",
                        "flag",
                        "priority",
                        "comments",
                    ],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [50912],
                }
            }
        
        self._icustay_key = "stay_id"
        self._hadm_key = "hadm_id"
        self._patient_key = "subject_id"

        self._determine_first_icu = "INTIME"
 
    def prepare_tasks(self, cohorts, spark=None, cached=False):
        labeled_cohorts = super().prepare_tasks(cohorts, spark, cached)
        if cached:
            return labeled_cohorts
       
        if self.diagnosis:
            logger.info("Start labeling cohorts for diagnosis prediction.")

            # define diagnosis prediction task
            diagnoses = pd.read_csv(os.path.join(self.data_dir, self.diagnosis_fname))

            diagnoses = self.icd10toicd9(diagnoses)

            ccs_dx = pd.read_csv(self.ccs_path)
            ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.strip()
            ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1]
            lvl1 = {
                x: int(y) - 1
                for _, (x, y) in ccs_dx[["'ICD-9-CM CODE'", "'CCS LVL 1'"]].iterrows()
            }

            diagnoses["diagnosis"] = diagnoses["icd_code_converted"].map(lvl1)

            diagnoses = diagnoses[
                (diagnoses["diagnosis"].notnull()) & (diagnoses["diagnosis"] != 14)
            ]
            diagnoses.loc[diagnoses["diagnosis"] >= 14, "diagnosis"] -= 1
            diagnoses = (
                diagnoses.groupby(self.hadm_key)["diagnosis"]
                .agg(lambda x: list(set(x)))
                .to_frame()
            )

            labeled_cohorts = labeled_cohorts.merge(
                diagnoses, on=self.hadm_key, how="inner"
            )

            logger.info("Done preparing diagnosis prediction for the given cohorts")

            self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")
        
        logger.info("Start labeling cohorts for clinical task prediction.")
 
        for clinical_task in [
            "creatinine",
            "platelets",
            "wbc",
            "hb",
            "bicarbonate",
            "sodium",
            "urine",
            "AKI"
        ]:
            
            horizons = self.__getattribute__(clinical_task)
            if horizons:
                labeled_cohorts = self.clinical_task(
                    labeled_cohorts,
                    clinical_task,
                    horizons
                )

        logger.info("Done preparing clinical task prediction for the given cohorts")
        
        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")
        
        return labeled_cohorts

    def make_compatible(self, icustays, spark):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patient_fname))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admission_fname))
        
        # prepare icustays according to the appropriate format
        icustays = icustays.rename(
            columns={
                "los": "LOS",
                "intime": "INTIME",
            }
        )
        admissions = admissions.rename(
            columns={
                "deathtime": "DEATHTIME",
                "admittime": "ADMITTIME",
            }
        )

        icustays = icustays[icustays["first_careunit"] == icustays["last_careunit"]]
        
        icustays["INTIME"] = pd.to_datetime(
            icustays["INTIME"], infer_datetime_format=True
        )

        icustays = icustays.merge(patients, on="subject_id", how="left")
        icustays["AGE"] = (
            icustays["INTIME"].dt.year
            - icustays["anchor_year"]
            + icustays["anchor_age"]
        )

        icustays = icustays.merge(
            admissions[
                [
                    self.hadm_key,
                    "DEATHTIME",
                    "ADMITTIME",
                ]
            ],
            how="left",
            on=self.hadm_key,
        )

        icustays["ADMITTIME"] = pd.to_datetime(
            icustays["ADMITTIME"], infer_datetime_format=True)
        icustays["INTIME"] = (
            pd.to_datetime(icustays["INTIME"], infer_datetime_format=True)
            - icustays["ADMITTIME"]
        ).dt.total_seconds() // 60
        icustays["DEATHTIME"] = (
            pd.to_datetime(icustays["DEATHTIME"], infer_datetime_format=True)
            - icustays["ADMITTIME"]
        ).dt.total_seconds() // 60

        # Extract Weight
        weight_itemid = 226512
        # Assuming F is imported from pyspark.sql, e.g., `from pyspark.sql import functions as F`
        # If not, this line will cause an error.
        from pyspark.sql import functions as F
        chartevents = spark.read.csv(
            os.path.join(self.data_dir, "icu/chartevents" + self.ext), header=True
        )
        weights = chartevents.filter(F.col("itemid") == weight_itemid).select(
            self.icustay_key, "valuenum"
        )
        # Take the first weight measurement (or average)
        weights = weights.groupby(self.icustay_key).agg(F.avg("valuenum").alias("weight"))
        weights = weights.toPandas()
        
        # Convert stay_id to int64 to match icustays dtype
        weights[self.icustay_key] = weights[self.icustay_key].astype('int64')
        
        icustays = icustays.merge(weights, on=self.icustay_key, how="left")

        if "weight" in icustays.columns:
            default_by_gender = {"M": 80.0, "F": 65.0}
            if "gender" in icustays.columns:
                for gender, default in default_by_gender.items():
                    mask = (icustays["gender"] == gender) & (icustays["weight"].isna())
                    icustays.loc[mask, "weight"] = default
            icustays["weight"] = icustays["weight"].fillna(72.5)

        return icustays

    def icd10toicd9(self, dx):
        #gem = pd.read_csv(self.gem_path)
        gem = pd.read_csv(self.gem_path, quotechar='"', quoting=csv.QUOTE_NONE, on_bad_lines='skip')

        # ADD : 모든 문자열 컬럼에 대해 양쪽 끝의 따옴표 제거
        gem.columns = [col.strip('"') for col in gem.columns]

        # 데이터프레임의 값에서도 따옴표 제거
        for col in gem.columns:
            if gem[col].dtype == object:
                gem[col] = gem[col].str.strip('"')
                
        dx_icd_10 = dx[dx["icd_version"] == 10]["icd_code"]

        unique_elem_no_map = set(dx_icd_10) - set(gem["icd10cm"])

        map_cms = dict(zip(gem["icd10cm"], gem["icd9cm"]))
        map_manual = dict.fromkeys(unique_elem_no_map, "NaN")

        for code_10 in map_manual:
            for i in range(len(code_10), 0, -1):
                tgt_10 = code_10[:i]
                if tgt_10 in gem["icd10cm"]:
                    tgt_9 = (
                        gem[gem["icd10cm"].str.contains(tgt_10)]["icd9cm"]
                        .mode()
                        .iloc[0]
                    )
                    map_manual[code_10] = tgt_9
                    break

        def icd_convert(icd_version, icd_code):
            if icd_version == 9:
                return icd_code

            elif icd_code in map_cms:
                return map_cms[icd_code]

            elif icd_code in map_manual:
                return map_manual[icd_code]
            else:
                logger.warn("WRONG CODE: " + icd_code)

        dx["icd_code_converted"] = dx.apply(
            lambda x: icd_convert(x["icd_version"], x["icd_code"]), axis=1
        )
        return dx

    def dialysis_filter(self, merge, timestamp):
        # Filter Dialysis at here to use abs timestamp & agg by patient_key
        # For Creatinine task, eliminate icus if patient went through dialysis treatment before pred_size timestamp
        # Filtering base on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/treatment/rrt.sql (Dialysis Active)
        dialysis_tables = self.task_itemids["dialysis"]["tables"]

        # Load dialysis related tables
        debug_row_limit = 200000 if self.debug else None
        chartevents = pd.read_csv(
            os.path.join(self.data_dir, "icu/chartevents" + self.ext),
            usecols=dialysis_tables["chartevents"]["include"],
            nrows=debug_row_limit,
        )
        inputevents = pd.read_csv(
            os.path.join(self.data_dir, "icu/inputevents" + self.ext),
            usecols=dialysis_tables["inputevents"]["include"],
            nrows=debug_row_limit,
        )
        procedureevents = pd.read_csv(
            os.path.join(self.data_dir, "icu/procedureevents" + self.ext),
            usecols=dialysis_tables["procedureevents"]["include"],
            nrows=debug_row_limit,
        )

        # Select only required columns
        chartevents = chartevents[dialysis_tables["chartevents"]["include"]]
        inputevents = inputevents[dialysis_tables["inputevents"]["include"]]
        procedureevents = procedureevents[dialysis_tables["procedureevents"]["include"]]

        # Filter dialysis related data
        ce = chartevents[
            ((chartevents["itemid"] == 225965) & (chartevents["value"] == "In use")) |
            (chartevents["itemid"].isin(dialysis_tables["chartevents"]["itemid"]["ce"]) & 
             chartevents["value"].notna())
        ]
        
        ie = inputevents[
            inputevents["itemid"].isin(dialysis_tables["inputevents"]["itemid"]["ie"]) &
            (inputevents["amount"] > 0)
        ]
        
        pe = procedureevents[
            procedureevents["itemid"].isin(dialysis_tables["procedureevents"]["itemid"]["pe"]) &
            procedureevents["value"].notna()
        ]

        # Extract dialysis times
        def dialysis_time(table, timecolumn):
            return table[[self.patient_key, timecolumn]].rename(
                columns={timecolumn: "_DIALYSIS_TIME"}
            )

        ce = dialysis_time(ce, "charttime")
        ie = dialysis_time(ie, "starttime")
        pe = dialysis_time(pe, "starttime")

        # Combine all dialysis records
        dialysis = pd.concat([ce, ie, pe])
        dialysis["_DIALYSIS_TIME"] = pd.to_datetime(dialysis["_DIALYSIS_TIME"])
        
        # Get earliest dialysis time per patient
        dialysis = dialysis.groupby(self.patient_key)["_DIALYSIS_TIME"].min().reset_index()

        # Join dialysis times and filter by timestamp
        merge = merge.merge(dialysis, on=self.patient_key, how="left")
        
        # Count patients before filtering
        initial_count = merge[self.patient_key].nunique()
        
        merge = merge[
            merge["_DIALYSIS_TIME"].isna() |
            (merge["_DIALYSIS_TIME"] > merge[timestamp])
        ]
        
        # Count patients after filtering
        final_count = merge[self.patient_key].nunique()
        
        # Print the number of excluded patients
        print(f"Number of patients excluded by dialysis filter: {initial_count - final_count}")
        
        merge = merge.drop(columns=["_DIALYSIS_TIME"])
        
        return merge
    
    def _normalize_units(self, table, task, value_col):
        if "valueuom" not in table.columns:
            return table

        uom = (
            table["valueuom"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        if task == "creatinine":
            umol_mask = uom.str.contains("umol")
            mg_l_mask = uom.str.contains("mg/l")
            table.loc[umol_mask, value_col] = table.loc[umol_mask, value_col] / 88.42
            table.loc[mg_l_mask, value_col] = table.loc[mg_l_mask, value_col] / 10.0
        elif task in ["platelets", "wbc"]:
            per_cell_mask = (
                uom.str.contains("cell")
                | (uom.str.contains("/ul") & ~uom.str.contains("10"))
            )
            table.loc[per_cell_mask, value_col] = table.loc[per_cell_mask, value_col] / 1000.0
        elif task == "hb":
            mmol_mask = uom.str.contains("mmol")
            gl_mask = uom.str.contains("g/l")
            table.loc[mmol_mask, value_col] = table.loc[mmol_mask, value_col] / 0.6206
            table.loc[gl_mask, value_col] = table.loc[gl_mask, value_col] / 10.0
        elif task == "urine":
            liter_mask = uom.str.contains("l") & ~uom.str.contains("ml")
            table.loc[liter_mask, value_col] = table.loc[liter_mask, value_col] * 1000.0

        return table
    
    
    def clinical_task(self, cohorts, task, horizons):
        print('task : ', task )
        fname = self.task_itemids[task]["fname"]
        timestamp = self.task_itemids[task]["timestamp"]
        excludes = self.task_itemids[task]["exclude"]
        code = self.task_itemids[task]["code"][0]
        value = self.task_itemids[task]["value"][0]
        itemid = self.task_itemids[task]["itemid"]
        
        start_time = time.time()  # Start timing
        table = pd.read_csv(os.path.join(self.data_dir, fname))

        end_time = time.time()  # End timing
        print(f"Time taken to read the CSV file: {round(end_time - start_time, 1)} seconds")
        
        table = table.drop(columns=excludes)
        table = table[table[code].isin(itemid) & table[value].notna()]
        table = self._normalize_units(table, task, value)
        
        if self.icustay_key in table.columns:
            table = table.drop(columns=[self.icustay_key])
        # Use Pandas merge instead of PySpark join
        
        merge = pd.merge(cohorts, table, on=self.hadm_key, how="inner")
        merge[timestamp] = pd.to_datetime(merge[timestamp], infer_datetime_format=True)
        merge["ADMITTIME"] = pd.to_datetime(merge["ADMITTIME"], infer_datetime_format=True)
        merge = merge.dropna(subset=[timestamp, "ADMITTIME"])
        
        if task in ["creatinine", "urine", "AKI"]:
            merge = self.dialysis_filter(merge, timestamp)
        
        merge[timestamp] = (merge[timestamp] - merge["ADMITTIME"]).dt.total_seconds() // 60
        
        if task == "AKI" and "baseline_creatinine" not in cohorts.columns:
            # Define the time window for baseline creatinine (7 days before INTIME)
            baseline_window = (
                (merge[timestamp] >= (merge["INTIME"] - 7*24*60)) & 
                (merge[timestamp] < (merge["INTIME"] + 12*60))
            )

            baseline_creatinine = merge[baseline_window].groupby(self.icustay_key).agg({value: "min"}
                            ).rename(columns={value: "baseline_creatinine"})
            merge = pd.merge(merge, baseline_creatinine, on=self.icustay_key, how="left")
            
            # Fallback: Use first measurement in ICU if baseline is missing
            icu_events = merge[merge[timestamp] >= merge["INTIME"]]
            if not icu_events.empty:
                first_icu = icu_events.sort_values(timestamp).groupby(self.icustay_key)[value].first()
                merge["baseline_creatinine"] = merge["baseline_creatinine"].fillna(merge[self.icustay_key].map(first_icu))
        
        merge = merge[merge[timestamp] >= (merge["INTIME"] + self.pred_size*60)]
        merge = merge.sort_values(by=[self.icustay_key, timestamp])
        
        for horizon in horizons:
            horizon_merge = merge[
                (merge[timestamp] < (merge["INTIME"] + (self.pred_size + horizon * 24) * 60)) &
                (merge[timestamp] >= (merge["INTIME"] + (self.pred_size + (horizon - 1) * 24) * 60))
            ]
            horizon_agg = horizon_merge.groupby(self.icustay_key).agg({value: "median"})

            if "baseline_creatinine" in merge.columns:
                horizon_agg = pd.merge(
                    horizon_agg,
                    merge[[self.icustay_key, "baseline_creatinine"]].drop_duplicates(),
                    on=self.icustay_key,
                    how="left"
                ).set_index(self.icustay_key)
            
            task_name = task + "_" + str(horizon)
            # Labeling
            if task == "platelets":
                horizon_agg[task_name] = pd.cut(
                    horizon_agg[value],
                    bins=[-float('inf'), 20, 50, 100, 150, float('inf')],
                    labels=[4, 3, 2, 1, 0],
                    include_lowest=True
                )

            elif task == "creatinine":
                horizon_agg[task_name] = pd.cut(
                    horizon_agg[value],
                    bins=[-float('inf'), 1.2, 2.0, 3.5, 5.0, float('inf')],
                    labels=[0, 1, 2, 3, 4],
                    include_lowest=True
                )

            elif task == "wbc":
                # NOTE: unit is K/ul (1000 cells per uL)
                horizon_agg[task_name] = pd.cut(
                    horizon_agg[value],
                    bins=[-float('inf'), 4, 12, float('inf')],
                    labels=[0, 1, 2],
                    include_lowest=True
                )

            elif task == "hb":
                horizon_agg[task_name] = pd.cut(
                    horizon_agg[value],
                    bins=[-float('inf'), 8, 10, 12, float('inf')],
                    labels=[0, 1, 2, 3],
                    include_lowest=True
                )

            elif task == "bicarbonate":
                horizon_agg[task_name] = pd.cut(
                    horizon_agg[value],
                    bins=[-float('inf'), 22, 29, float('inf')],
                    labels=[0, 1, 2],
                    include_lowest=True
                )

            elif task == "sodium":
                horizon_agg[task_name] = pd.cut(
                    horizon_agg[value],
                    bins=[-float('inf'), 135, 145, float('inf')],
                    labels=[0, 1, 2],
                    include_lowest=True
                )
            elif task == "urine":
                if "weight" in merge.columns:
                    # Merge weight into horizon_agg if not present (it might be in cohorts, but horizon_agg is grouped)
                    # Actually horizon_agg is grouped by icustay_key, so we can map weight
                    weight_map = merge[[self.icustay_key, "weight"]].drop_duplicates().set_index(self.icustay_key)["weight"]
                    horizon_agg["weight"] = horizon_agg.index.map(weight_map)
                    horizon_agg[value] = horizon_agg[value] / horizon_agg["weight"]
                
                horizon_agg[task_name] = pd.cut(
                    horizon_agg[value],
                    bins=[-float('inf'), 0.05, 0.1, 0.3, 0.5, float('inf')],
                    labels=[4, 3, 2, 1, 0],
                    include_lowest=True
                )
                
            elif task == "AKI":
                horizon_agg[task_name] = pd.cut(
                    (horizon_agg[value] - horizon_agg["baseline_creatinine"]) / horizon_agg["baseline_creatinine"],
                    bins=[-float('inf'), 0.3, 1.5, 2.0, 3.0, float('inf')],
                    labels=[0, 1, 2, 3, 4],
                    include_lowest=True
                )
                # 안전장치: 범위를 벗어나거나 변환 실패 시 -1로 마스킹
                horizon_agg[task_name] = pd.to_numeric(horizon_agg[task_name], errors="coerce")
                horizon_agg[task_name] = horizon_agg[task_name].where(
                    horizon_agg[task_name].isin([0, 1, 2, 3, 4]), -1
                )
            
            cohorts = pd.merge(cohorts,
                horizon_agg[task_name],
                on=self.icustay_key,
                how="left"
            )
            
            if (f"AKI_{horizon}" in cohorts.columns 
                and f"urine_{horizon}" in cohorts.columns):
                # Use pandas max with skipna=False to ensure NaN is returned if both are NaN
                cohorts[f"AKI_{horizon}"] = cohorts[[f"AKI_{horizon}", f"urine_{horizon}"]].max(axis=1, skipna=False)
                logger.info("AKI stage is combined from creatinine and urine output stages.")

        return cohorts

    def infer_data_extension(self) -> str:
        if (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*.csv.gz"))) == 22
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*.csv.gz"))) == 8
        ):
            ext = ".csv.gz"
        elif (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*.csv"))) == 22
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*.csv"))) == 8
        ):
            ext = ".csv"
        else:
            raise AssertionError(
                "Provided data directory is not correct. Please check if --data is correct. "
                "--data: {}".format(self.data_dir)
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext
