# The folder indicated here must contain the following files:

import pandas as pd
import os
from myutils import FILECONSTANTS as FC
import myutils.utils as ut
import numpy as np

COLS = ["bus", "latitude", "longitude", "area", "baskv", "bus_name"]
fn = f"{FC.INPUTDIR}BUSES/MyIn_BusLocation_with_state.parquet"
BUSES = pd.read_parquet(fn)[COLS]
BUSES = ut.identify_osg_buses_types(BUSES)

COL_RENAME = {
    " Area Name": ("area_name", str),
    " Area Num": ("area", np.int64),
    "Angle(deg)": ("vangle", np.float64),
    "B-Zero (pu)": ("bzero", np.float64),
    "Base kV": ("baskv", np.float64),
    "Bus  Name": ("bus_name", str),
    "Bus  Number": ("bus", np.int64),
    "CCC Accel": ("dccccaccel", np.float64),
    "CCC Itmax": ("dccccitmax", np.float64),
    "CF": ("cf", np.float64),
    "Charging B (pu)": ("line_b", np.float64),
    "Code": ("code", np.int64),
    "Control Mode": ("cont_mode", str),
    "Dcvmin (kV)": ("dcvmin", np.float64),
    "Delti (pu)": ("delti", np.float64),
    "Distributed Gen (MW)": ("dgmw", np.float64),
    "Distributed Gen (Mvar)": ("dbmvar", np.float64),
    "Distributed Gen Mode": ("dg_mode", np.float64),
    "Emergency Vmax(pu)": ("em_vmax", np.float64),
    "Emergency Vmin(pu)": ("em_vmin", np.float64),
    "Fraction 1": ("fr_1", np.float64),
    "Fraction 2": ("fr_2", np.float64),
    "Fraction 3": ("fr_3", np.float64),
    "Fraction 4": ("fr_4", np.float64),
    "From Bus  Name": ("fr_bus_name", np.float64),
    "From Bus  Number": ("from", np.float64),
    "Grounding flag": ("grounding_flag", np.float64),
    "IPload (MW)": ("ipload", np.float64),
    "IQload (Mvar)": ("iqload", np.float64),
    "Id": ("id", str),
    "In Service": ("status", np.int64),
    "Interruptible": ("interruptible", np.float64),
    "Inverter": ("to", np.int64),
    "Length": ("length", np.float64),
    "Line B From (pu)": ("line_b_fr", np.float64),
    "Line B To (pu)": ("line_b_to", np.float64),
    "Line G From (pu)": ("line_g_fr", np.float64),
    "Line G To (pu)": ("line_g_to", np.float64),
    "Line Name": ("line_name", str),
    "Line R (pu)": ("line_r", np.float64),
    "Line X (pu)": ("line_x", np.float64),
    "MOV Rated Current (kA)": ("mov_rated_current", np.float64),
    "Mbase (MVA)": ("mbase", np.float64),
    "Metered": ("metered", np.float64),
    "Name": ("name", str),
    "Normal Vmax(pu)": ("normal_vmax", np.float64),
    "Normal Vmin(pu)": ("normal_vmin", np.float64),
    "Owner 1": ("owner_1", np.float64),
    "Owner 2": ("owner_2", np.float64),
    "Owner 3": ("owner_3", np.float64),
    "Owner 4": ("owner_4", np.float64),
    "Owner Name": ("owner_name", str),
    "Owner Num": ("owner", str),
    "PGen (MW)": ("pgen", np.float64),
    "PMax (MW)": ("pmax", np.float64),
    "PMin (MW)": ("pmin", np.float64),
    "PNeg (MW)": ("pneg", np.float64),
    "PZero (MW)": ("pzero", np.float64),
    "Pload (MW)": ("pload", np.float64),
    "Protection Mode": ("prot_mode", str),
    "QGen (Mvar)": ("qgen", np.float64),
    "QMax (Mvar)": ("qmax", np.float64),
    "QMin (Mvar)": ("qmin", np.float64),
    "QNeg (Mvar)": ("qneg", np.float64),
    "QZero (Mvar)": ("qzero", np.float64),
    "Qload (Mvar)": ("qload", np.float64),
    "R-Zero (pu)": ("rzero", np.float64),
    "RATE1 (I as MVA)": ("rate1", np.float64),
    "RATE10 (I as MVA)": ("rate10", np.float64),
    "RATE11 (I as MVA)": ("rate11", np.float64),
    "RATE12 (I as MVA)": ("rate12", np.float64),
    "RATE2 (I as MVA)": ("rate2", np.float64),
    "RATE3 (I as MVA)": ("rate3", np.float64),
    "RATE4 (I as MVA)": ("rate4", np.float64),
    "RATE5 (I as MVA)": ("rate5", np.float64),
    "RATE6 (I as MVA)": ("rate6", np.float64),
    "RATE7 (I as MVA)": ("rate7", np.float64),
    "RATE8 (I as MVA)": ("rate8", np.float64),
    "RATE9 (I as MVA)": ("rate9", np.float64),
    "Rcmp-Ohm (ohms)": ("rcmp_ohm", np.float64),
    "Rdc (ohms)": ("rdc", np.float64),
    "Rectifier": ("from", np.int64),
    "Regulated Bus Number": ("reg_bus", np.int64),
    "Scalable": ("scalable", np.int64),
    "Section Number": ("section", str),
    "Setval (amps or MW)": ("setval", np.float64),
    "Substation  Number": ("substation", np.float64),
    "Term Node Name  ": ("term_node_name", str),
    "Term Node Name (From)": ("term_node_name_fr", str),
    "Term Node Name (To)": ("term_node_name_to", str),
    "Term Node Num  ": ("term_node_num", np.int64),
    "Term Node Num (From)": ("term_node_num_fr", np.int64),
    "Term Node Num (To)": ("term_node_num_to", np.int64),
    "To Bus  Name": ("to_bus_name", np.int64),
    "To Bus  Number": ("to", np.int64),
    "Type": ("type", str),
    "VSched (kV)": ("vsched", np.float64),
    "Vcmode (kV)": ("vcmode", np.float64),
    "Voltage(pu)": ("volt", np.float64),
    "X-Zero (pu)": ("xzero", np.float64),
    "YPload (MW)": ("ypload", np.float64),
    "YQload (Mvar)": ("yqload", np.float64),
    "Zero Seq B From (pu)": ("zero_seq_b_fr", np.float64),
    "Zero Seq B To (pu)": ("zero_seq_b_to", np.float64),
    "Zero Seq G From (pu)": ("zero_seq_g_fr", np.float64),
    "Zero Seq G To (pu)": ("zero_seq_g_to", np.float64),
    "Zone Name": ("zone_name", str),
    "Zone Num": ("zone", np.int64),
    "category": ("category", str),
}

SELCOLS = {
    "PSSE_Buses": ["bus", "bus_name", "baskv", "area", "zone", "code"],
    "PSSE_Loads": ["bus", "id", "area", "status", "code", "pload", "qload"],
    "PSSE_DCLines": [
        "from",
        "to",
        "cont_mode",
        "delti",
        "setval",
        "vsched",
    ],
    "PSSE_Gens_Tech_latest": [
        "bus",
        "area",
        "code",
        "status",
        "pgen",
        "qgen",
        "mbase",
        "pmax",
        "pmin",
        "qmax",
        "qmin",
        "category",
    ],
    "PSSE_Lines": ["from", "to", "id", "line_x", "status"],
}

# this must be moved to create_database.py
READEXCEL = False
if READEXCEL:
    fn = f"{FC.INPUTDIR}/PSSE/In_PSSEData.xlsx"
    sheets = {
        "PSSE_Buses": [0],
        "PSSE_Gens_Tech_latest": [0],
        "PSSE_Loads": [0],
        "PSSE_Lines": [0],
        "PSSE_DCLines": [0],
    }
    PSSEDFS = {sh: pd.ExcelFile(fn).parse(
        sh, header=sheets[sh]) for sh in sheets}
    for sh, df in PSSEDFS.items():
        rename_dict = {k: v[0]
                       for k, v in COL_RENAME.items() if k in df.columns}
        df.rename(columns=rename_dict, inplace=True)
        df.columns = df.columns.map(str)
        df = df[SELCOLS[sh]]

        for _, (col, dtype) in COL_RENAME.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype, errors="ignore")
        if "status" in df.columns:
            df = df[df["status"] == 1]
        if "code" in df.columns:
            df = df[df["code"] != 4]
        try:
            df.to_parquet(f"{FC.INPUTDIR}PSSE/{sh}.parquet")
        except:
            print(f"ERRROR Could not save {sh} to parquet ERROR")

else:
    PSSEDFS = {}
    folder_path = f"{FC.INPUTDIR}/PSSE/"
    fns = [
        file.split(".")[0]
        for file in os.listdir(folder_path)
        if (file.endswith(".parquet"))
    ]
    for fn in fns:
        PSSEDFS[fn] = pd.read_parquet(f"{folder_path}{fn}.parquet")


PSSELOADS = PSSEDFS["PSSE_Loads"]
FULLBUSINFOMAP = PSSEDFS["PSSE_Buses"][["bus", "baskv", "area"]]
PSSEGENS = PSSEDFS["PSSE_Gens_Tech_latest"]
POIS = pd.read_excel(FC.INPUTDIR + "POIS/In_POIS.xlsx")["bus"].values
