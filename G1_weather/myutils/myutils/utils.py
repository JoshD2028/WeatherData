import pandas as pd
from myutils import FILECONSTANTS as FC

import numpy as np


def identify_osg_buses_types(df):
    # df["ord"] = np.log10(df["bus"]).astype(np.int64)
    df["ord"] = df["bus"].astype(str).str.len()
    # this is backbone
    df.loc[df["ord"] == 5, "isbb"] = 1

    # this is energy islands
    df.loc[df["ord"] == 3, "isei"] = 1

    df.loc[df["ord"] == 4, "islf"] = 1

    df[["area", "isbb", "isei", "islf"]] = df[["area", "isbb", "isei", "islf"]].fillna(
        0
    )
    df[["bus", "area", "isbb", "islf"]] = df[["bus", "area", "isbb", "islf"]].astype(
        np.int64
    )

    df = df.drop(columns="ord")

    return df


def identify_osg_types(df):
    """identify reach circuits and landfall intconnects"""
    df["ord_from"] = np.log10(df["from"]).astype(np.int64)
    df["ord_to"] = np.log10(df["to"]).astype(np.int64)
    df.loc[(df["ord_from"].isin([2, 4])) & (
        df["ord_to"].isin([2, 4])), "isbb"] = 1
    df.loc[
        ((df["ord_from"].isin([2, 4])) & (df["ord_to"] == 3))
        | ((df["ord_to"].isin([2, 4])) & (df["ord_from"] == 3)),
        "islf",
    ] = 1
    df.loc[
        ((df["ord_from"] == 5) & (df["ord_to"] == 3))
        | ((df["ord_to"] == 5) & (df["ord_from"] == 3))
        | ((df["ord_from"] == 5) & (df["ord_to"] == 5)),
        "isrc",
    ] = 1

    df = df.drop(columns=["ord_from", "ord_to"])
    df = df.fillna(0)
    df[["isrc", "islf"]] = df[["isrc", "islf"]].fillna(0)
    # all columns except length must be of type int64
    df[["from", "to", "isbb", "islf", "isrc"]] = df[
        ["from", "to", "isbb", "islf", "isrc"]
    ].astype(np.int64)
    return df


def add_iso(df, AREAS=FC.load_yaml(f"{FC.INPUTDIR}/AREAS/areas.yaml"), areacol="area"):
    if "ALLAREAS" in AREAS.keys():
        del AREAS["ALLAREAS"]
    for i, (k, v) in enumerate(AREAS.items()):
        # print(i, k, v)
        df.loc[df[areacol].isin(v), "iso"] = k
    df["iso"] = df["iso"].fillna(value="OUT")
    return df


def add2lines_businfo(
    df, bus, lefton1="from", righton1="bus", lefton2="to", righton2="bus"
):
    """add to df the areas; df is the branches"""

    # df = pd.merge(
    #     pd.merge(df, bus, left_on=lefton1, right_on=righton1, how='left').drop(
    #         columns=[righton1]
    #     ),
    #     bus,
    #     left_on=lefton2,
    #     right_on=righton2,
    #     suffixes=('_i', '_j'),
    #     how='left',
    # ).drop(columns=[righton2])

    bus = bus[[righton1, "name", "baskv", "area", "zone", "isin"]].rename(
        columns={"name": "busname"}
    )
    df = pd.merge(
        pd.merge(df, bus, left_on=lefton1, right_on=righton1, how="left"),
        bus.rename(columns={righton2: lefton2}),
        on=lefton2,
        suffixes=("_i", "_j"),
        how="left",
    )
    return df


def add2bus_businfo(df, busmap, lefton="bus", righton="bus"):
    """add to buses voltage level and area name"""
    if isinstance(df, list):
        return busmap.loc[busmap[righton].isin(df), :]
    df = pd.merge(
        df,
        busmap,
        left_on=lefton,
        right_on=righton,
        how="left",
        suffixes=("_i", "_j"),
    )

    if lefton != righton:
        df = df.drop(columns=[righton])

    return df


def sort_from_to(df, fb="ibus", tb="jbus"):
    """return sorted from to; must contain ibus, jbus"""
    df["fromto"] = tuple(zip(df[fb], df[tb]))
    df["fromto"] = df["fromto"].apply(lambda x: sorted(x))
    df[[fb, tb]] = pd.DataFrame(df["fromto"].tolist(), index=df.index)
    df2 = df.drop(columns="fromto")
    return df2, df
