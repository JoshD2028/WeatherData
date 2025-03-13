"""
Power System Data Processing Utilities

This module provides utility functions for data processing, reduction, and load mapping in a power system analysis.
"""

import numpy as np
import pandas as pd
import yaml
from myutils import utils


def fix_buses(bus_data: pd.DataFrame, internals: list) -> tuple:
    """
    Fix the buses before selecting the retained buses.

    This function modifies the 'isin' column of the bus DataFrame based on the 'area' values.
    Buses with 'area' in the 'internals' list are marked as retained (isin = 1), while others are marked as not retained (isin = 0).

    Args:
        bus_data: The DataFrame containing bus data.
        internals: A list of internally retained areas.

    Returns:
        A tuple of two DataFrames: (fixed_bus_data, retained_bus_data).

    Example:
        >>> bus_data = pd.DataFrame(...)  # create or load the bus DataFrame
        >>> internals = ["Area1", "Area2"]
        >>> fixed_buses, retained_buses = fix_buses(bus_data, internals)
    """
    bus_data.loc[bus_data['area'].isin(internals), 'isin'] = 1
    bus_data.loc[~bus_data['area'].isin(internals), 'isin'] = 0

    nonisland_buses = bus_data.loc[
        bus_data['ide'] != 4,
        ['ibus', 'name', 'baskv', 'ide', 'area', 'zone', 'isin'],
    ]
    fixed_buses = bus_data.loc[
        :, ['ibus', 'name', 'baskv', 'ide', 'area', 'zone', 'isin']
    ]

    return fixed_buses, nonisland_buses


def fix_load(load_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the loads before selecting the retained buses.

    This function filters the load DataFrame by removing rows with 'stat' value of 0.

    Args:
        load_data: The DataFrame containing load data.

    Returns:
        The filtered load DataFrame.

    Example:
        >>> load_data = pd.DataFrame(...)  # create or load the load DataFrame
        >>> fixed_load = fix_load(load_data)
    """
    # selecting the load that is on
    fixed_load = load_data.loc[
        load_data['stat'] != 0,
        ['ibus', 'loadid', 'stat', 'area', 'zone', 'pl', 'ql'],
    ]

    return fixed_load


def fix_gens(gens_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the generators before selecting the retained buses.

    This function filters the generators DataFrame by removing rows with 'stat' value of 0.

    Args:
        gens_data: The DataFrame containing generator data.

    Returns:
        The filtered generator DataFrame.

    Example:
        # create or load the generator DataFrame
        >>> gens_data = pd.DataFrame(...)
        >>> fixed_gens = fix_gens(gens_data)
    """
    # select the generators that are on
    fixed_gens = gens_data.loc[
        gens_data['stat'] != 0,
        ['ibus', 'machid', 'pg', 'qg', 'qt', 'qb', 'vs', 'ireg', 'mbase'],
    ]

    return fixed_gens


def fix_tt_dc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate P, Qmin, and Qmax for rectifiers and inverters.

    Args:
        df (pandas.DataFrame): DataFrame containing TT DC line information.

    Returns:
        pandas.DataFrame: DataFrame with calculated parameters for TT DC lines.
    """
    # Calculate P for rectifiers and inverters
    df.loc[df['mdc'] == 1, 'Pmw'] = df.loc[df['mdc'] == 1, 'setvl']
    df.loc[df['mdc'] == 2, 'Pmw'] = (
        df.loc[df['mdc'] == 2, 'setvl']
        * df.loc[df['mdc'] == 2, 'vschd']
        / 1000
    )
    df.loc[~df['mdc'].isin([1, 2]), 'Pmw'] = 0
    df.loc[df['Pmw'] < 0, 'Pmw'] *= -1

    # Calculate Qmin and Qmax for rectifiers and inverters
    df['Qminr'] = df['Pmw'] * df['anmnr']
    df['Qmini'] = df['Pmw'] * df['anmni']

    df['Qmaxr'] = df['Pmw'] * np.tan(
        np.arccos(0.5 * (np.cos(np.deg2rad(df['anmxr'])) + np.cos(np.pi / 3)))
    )
    df['Qmaxi'] = df['Pmw'] * np.tan(
        np.arccos(0.5 * (np.cos(np.deg2rad(df['anmxi'])) + np.cos(np.pi / 3)))
    )

    df.loc[df['Qminr'] < 0, 'Qminr'] *= -1
    df.loc[df['Qmini'] < 0, 'Qmini'] *= -1
    df.loc[df['Qmaxr'] < 0, 'Qmaxr'] *= -1
    df.loc[df['Qmaxi'] < 0, 'Qmaxi'] *= -1

    # Set status for TT DC lines
    df.loc[df['mdc'] == 0, 'stat'] = 0
    df.loc[df['mdc'] != 0, 'stat'] = 1

    # Create DataFrame for TT DC lines
    tt_dc_lines = df.loc[df['stat'] != 0, ['ipi', 'ipr', 'Pmw']].rename(
        columns={'ipi': 'ibus', 'ipr': 'jbus', 'Pmw': 'rate1'}
    )
    tt_dc_lines['rate2'] = tt_dc_lines['rate1']
    tt_dc_lines['rate3'] = tt_dc_lines['rate1']
    tt_dc_lines['ckt'] = tt_dc_lines.index
    tt_dc_lines['isdc'] = 1
    tt_dc_lines['b'] = 0

    return tt_dc_lines


def fix_branch(branches_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the branches before selecting the retained buses.

    This function filters the branch DataFrame by removing rows with 'stat' value of 0.
    It also renames the 'rpu', 'xpu', and 'bpu' columns to 'r', 'x', and 'b', respectively.
    Additionally, it adds an 'isac' column with a value of 1.

    Args:
        branches_data: The DataFrame containing branch data.

    Returns:
        The filtered and modified branch DataFrame.

    Example:
        # create or load the branch DataFrame
        >>> branches_data = pd.DataFrame(...)
        >>> fixed_branches = fix_branch(branches_data)
    """
    # select the branches that are on; rename some columns
    fixed_branches = branches_data.loc[
        branches_data['stat'] != 0,
        [
            'ibus',
            'jbus',
            'ckt',
            'rpu',
            'xpu',
            'bpu',
            'rate1',
            'rate2',
            'rate3',
        ],
    ].rename(columns={'rpu': 'r', 'xpu': 'x', 'bpu': 'b'})

    # these branches are AC lines
    fixed_branches['isac'] = 1

    return fixed_branches


def fix_trans2w(trans2w_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the two-winding transformers before selecting the retained buses.

    This function filters the two-winding transformer DataFrame by removing rows with 'stat' value of 0.
    It also renames the 'r1_2', 'x1_2', 'wdg1rate1', 'wdg1rate2', and 'wdg1rate3' columns to 'r', 'x', 'rate1', 'rate2', and 'rate3', respectively.
    Additionally, it adds a 'b' column with a value of 0 and an 'is2w' column with a value of 1.

    Args:
        trans2w_data: The DataFrame containing two-winding transformer data.

    Returns:
        The filtered and modified two-winding transformer DataFrame.

    Example:
        # create or load the two-winding transformer DataFrame
        >>> trans2w_data = pd.DataFrame(...)
        >>> fixed_trans2w = fix_trans2w(trans2w_data)
    """
    fixed_trans2w = trans2w_data.loc[
        trans2w_data['stat'] != 0,
        [
            'ibus',
            'jbus',
            'ckt',
            'r1_2',
            'x1_2',
            'wdg1rate1',
            'wdg1rate2',
            'wdg1rate3',
        ],
    ].rename(
        columns={
            'r1_2': 'r',
            'x1_2': 'x',
            'wdg1rate1': 'rate1',
            'wdg1rate2': 'rate2',
            'wdg1rate3': 'rate3',
        }
    )

    fixed_trans2w['b'] = 0
    fixed_trans2w['is2w'] = 1

    return fixed_trans2w


def fix_trans3w(trans3w_data: pd.DataFrame) -> pd.DataFrame:
    """
    XXX TODO XXX the impedance of 3w transformers need to be changed according to their code
    Fix the three-winding transformers before selecting the retained buses.

    This function filters the three-winding transformer DataFrame by removing rows with 'stat' value of 0.
    It also performs some transformations to create a modified DataFrame with the desired columns.
    The resulting DataFrame includes the columns 'ibus', 'jbus', 'ckt', 'r', 'x', 'rate1', 'rate2', 'rate3',
    'is3w', and 'b'.

    Args:
        trans3w_data: The DataFrame containing three-winding transformer data.

    Returns:
        The filtered and modified three-winding transformer DataFrame.

    Example:
        # create or load the three-winding transformer DataFrame
        >>> trans3w_data = pd.DataFrame(...)
        >>> fixed_trans3w = fix_trans3w(trans3w_data)
    """
    trans3w_preprocessed = trans3w_data.loc[
        trans3w_data['stat'] != 0,
        [
            'ibus',
            'jbus',
            'kbus',
            'ckt',
            'cw',
            'cz',
            'name',
            'stat',
            'vecgrp',
            'r1_2',
            'x1_2',
            'sbase1_2',
            'r2_3',
            'x2_3',
            'sbase2_3',
            'r3_1',
            'x3_1',
            'sbase3_1',
            'wdg1rate1',
            'wdg1rate2',
            'wdg1rate3',
            'wdg2rate1',
            'wdg2rate2',
            'wdg2rate3',
            'wdg3rate1',
            'wdg3rate2',
            'wdg3rate3',
        ],
    ]
    trans3w_preprocessed.loc[trans3w_preprocessed['stat'] == 1, 'nl'] = 3
    trans3w_preprocessed.loc[trans3w_preprocessed['stat'] == 0, 'nl'] = 0
    trans3w_preprocessed.loc[
        trans3w_preprocessed['stat'].isin([2, 3, 4]), 'nl'
    ] = 2
    print(f"total number of lines {trans3w_preprocessed['nl'].sum()}")
    trans3w_preprocessed = trans3w_preprocessed.drop(columns='nl')

    t3w = []
    for (
        ibus,
        jbus,
        kbus,
        ckt,
        cw,
        cz,
        name,
        stat,
        vecgrp,
        r1_2,
        x1_2,
        sbase1_2,
        r2_3,
        x2_3,
        sbase2_3,
        r3_1,
        x3_1,
        sbase3_1,
        wdg1rate1,
        wdg1rate2,
        wdg1rate3,
        wdg2rate1,
        wdg2rate2,
        wdg2rate3,
        wdg3rate1,
        wdg3rate2,
        wdg3rate3,
    ) in trans3w_preprocessed.values:
        if stat == 1:
            t3w.append(
                [
                    ibus,
                    jbus,
                    ckt,
                    cw,
                    cz,
                    name,
                    stat,
                    vecgrp,
                    r1_2,
                    x1_2,
                    sbase1_2,
                    wdg1rate1,
                    wdg1rate2,
                    wdg1rate3,
                ]
            )
            t3w.append(
                [
                    jbus,
                    kbus,
                    ckt,
                    cw,
                    cz,
                    name,
                    stat,
                    vecgrp,
                    r2_3,
                    x2_3,
                    sbase2_3,
                    wdg2rate1,
                    wdg2rate2,
                    wdg2rate3,
                ]
            )
            t3w.append(
                [
                    ibus,
                    kbus,
                    ckt,
                    cw,
                    cz,
                    name,
                    stat,
                    vecgrp,
                    r3_1,
                    x3_1,
                    sbase3_1,
                    wdg3rate1,
                    wdg3rate2,
                    wdg3rate3,
                ]
            )
        elif stat == 2:  # only winding 2 is out
            t3w.append(
                [
                    ibus,
                    jbus,
                    ckt,
                    cw,
                    cz,
                    name,
                    stat,
                    vecgrp,
                    r1_2,
                    x1_2,
                    sbase1_2,
                    wdg1rate1,
                    wdg1rate2,
                    wdg1rate3,
                ]
            )
            t3w.append(
                [
                    ibus,
                    kbus,
                    ckt,
                    cw,
                    cz,
                    name,
                    stat,
                    vecgrp,
                    r3_1,
                    x3_1,
                    sbase3_1,
                    wdg3rate1,
                    wdg3rate2,
                    wdg3rate3,
                ]
            )
        elif stat == 3:  # only winding 3 is out: add 1 and 2
            t3w.append(
                [
                    ibus,
                    jbus,
                    ckt,
                    cw,
                    cz,
                    name,
                    stat,
                    vecgrp,
                    r1_2,
                    x1_2,
                    sbase1_2,
                    wdg1rate1,
                    wdg1rate2,
                    wdg1rate3,
                ]
            )
            t3w.append(
                [
                    jbus,
                    kbus,
                    ckt,
                    cw,
                    cz,
                    name,
                    stat,
                    vecgrp,
                    r2_3,
                    x2_3,
                    sbase2_3,
                    wdg2rate1,
                    wdg2rate2,
                    wdg2rate3,
                ]
            )
        elif stat == 1:  # only winding 1 is out: add 2 and 3
            t3w.append(
                [
                    jbus,
                    kbus,
                    ckt,
                    cw,
                    cz,
                    name,
                    stat,
                    vecgrp,
                    r2_3,
                    x2_3,
                    sbase2_3,
                    wdg2rate1,
                    wdg2rate2,
                    wdg2rate3,
                ]
            )
            t3w.append(
                [
                    ibus,
                    kbus,
                    ckt,
                    cw,
                    cz,
                    name,
                    stat,
                    vecgrp,
                    r3_1,
                    x3_1,
                    sbase3_1,
                    wdg3rate1,
                    wdg3rate2,
                    wdg3rate3,
                ]
            )

    trans3w_fixed = pd.DataFrame(
        data=t3w,
        columns=[
            'ibus',
            'jbus',
            'ckt',
            'cw',
            'cz',
            'name',
            'stat',
            'vecgrp',
            'r',
            'x',
            'sbase',
            'rate1',
            'rate2',
            'rate3',
        ],
    )[['ibus', 'jbus', 'ckt', 'r', 'x', 'rate1', 'rate2', 'rate3']]

    trans3w_fixed['is3w'] = 1
    trans3w_fixed['b'] = 0

    return trans3w_fixed


def calculate_active_reactive_power(df):
    """
    Calculate active power (p), minimum reactive power (q_min),
    and maximum reactive power (q_max) for rectifier and inverter.

    Args:
        df (pandas.DataFrame): Input DataFrame containing required columns.

    Returns:
        pandas.DataFrame: DataFrame with calculated values for p, q_min, and q_max.
    """
    # Set active power (p) based on mdc values
    df.loc[df['mdc'] == 1, 'active_power'] = df.loc[df['mdc'] == 1, 'setvl']
    df.loc[df['mdc'] == 2, 'active_power'] = (
        df.loc[df['mdc'] == 2, 'setvl']
        * df.loc[df['mdc'] == 2, 'vschd']
        / 1000
    )
    df.loc[~(df['mdc'].isin([1, 2])), 'active_power'] = 0
    df.loc[df['active_power'] < 0, 'active_power'] = (
        -1 * df.loc[df['active_power'] < 0, 'active_power']
    )

    # Calculate minimum reactive power (q_min) and maximum reactive power (q_max) for rectifiers and inverters
    df.loc[:, 'q_min_r'] = df['active_power'] * df['anmnr']
    df.loc[:, 'q_min_i'] = df['active_power'] * df['anmni']

    df.loc[:, 'q_max_r'] = df['active_power'] * np.tan(
        np.arccos(0.5 * (np.cos(np.deg2rad(df['anmxr'])) + np.cos(np.pi / 3)))
    )
    df.loc[:, 'q_max_i'] = df['active_power'] * np.tan(
        np.arccos(0.5 * (np.cos(np.deg2rad(df['anmxi'])) + np.cos(np.pi / 3)))
    )

    # Make sure q_min and q_max are positive
    df.loc[df['q_min_r'] < 0, 'q_min_r'] = (
        -1 * df.loc[df['q_min_r'] < 0, 'q_min_r']
    )
    df.loc[df['q_min_i'] < 0, 'q_min_i'] = (
        -1 * df.loc[df['q_min_i'] < 0, 'q_min_i']
    )
    df.loc[df['q_max_r'] < 0, 'q_max_r'] = (
        -1 * df.loc[df['q_max_r'] < 0, 'q_max_r']
    )
    df.loc[df['q_max_i'] < 0, 'q_max_i'] = (
        -1 * df.loc[df['q_max_i'] < 0, 'q_max_i']
    )

    # Set stat values based on mdc
    df.loc[df['mdc'] == 0, 'stat'] = 0
    df.loc[df['mdc'] != 0, 'stat'] = 1

    return df


def calculate_bus_capacity(all_branches, bus_on):
    """
    Calculate the capacity of each substation.

    Args:
        all_branches (pandas.DataFrame): DataFrame containing all branches.
        bus_on (pandas.DataFrame): DataFrame containing bus information.

    Returns:
        pandas.DataFrame: DataFrame with calculated substation capacities.
    """
    # Calculate counts for ibus and jbus
    ic = (
        all_branches.groupby(['ibus'])['jbus']
        .count()
        .reindex(bus_on['ibus'])
        .fillna(value=0)
    )
    jc = (
        all_branches.groupby(['jbus'])['ibus']
        .count()
        .reindex(bus_on['ibus'])
        .fillna(value=0)
    )
    ijc = ic + jc
    ijc.name = 'count'
    ijc = ijc.reset_index()

    # Calculate sum and max rates for ibus and jbus
    ibus = (
        all_branches.groupby(['ibus'])[['rate2']]
        .agg({'rate2': ['sum', 'max']})
        .reindex(bus_on['ibus'])
        .fillna(value=0)['rate2']
    )
    jbus = (
        all_branches.groupby(['jbus'])[['rate2']]
        .agg({'rate2': ['sum', 'max']})
        .reindex(bus_on['ibus'])
        .fillna(value=0)['rate2']
    )

    # Calculate substation capacity
    subcap = pd.DataFrame()
    subcap['sum'] = ibus['sum'] + jbus['sum']
    subcap['max'] = np.maximum(ibus['max'], jbus['max'])
    subcap['cap'] = subcap['sum'] - subcap['max']
    subcap = subcap.sort_values(by=['cap'], ascending=False)
    buscap = pd.merge(bus_on, subcap[['cap']].reset_index(), on='ibus')
    buscapc = pd.merge(buscap, ijc, on='ibus')

    return buscapc


def get_generator_buses(
    generators, buscapc, trans2wdf, internals, gen_cap_threshold
):
    """
    Get buses connected to generators.

    Args:
        generators (pandas.DataFrame): DataFrame containing generator information.
        buscapc (pandas.DataFrame): DataFrame containing bus capacity information.
        trans2wdf (pandas.DataFrame): DataFrame containing transformer information.
        internals (list): List of internal areas.
        gen_cap_threshold (float): Generator capacity threshold.

    Returns:
        list: List of generator buses.
    """
    # Merge generators with buscapc to get area information
    gens_area = pd.merge(
        generators[['ibus', 'mbase']],
        buscapc[['ibus', 'area']],
        on='ibus',
        how='left',
    )

    # Get low-side generator buses
    gens_100mw = gens_area.loc[
        (gens_area['area'].isin(internals))
        & (gens_area['mbase'] > gen_cap_threshold)
    ]
    gens_buses = gens_100mw['ibus'].tolist()

    # Add voltage base to buses
    t2w = trans2wdf[['ibus', 'jbus']]
    t2w = pd.merge(
        pd.merge(t2w, buscapc[['ibus', 'baskv']], on='ibus').rename(
            columns={'baskv': 'kvi'}
        ),
        buscapc[['ibus', 'baskv']].rename(columns={'ibus': 'jbus'}),
        on='jbus',
    ).rename(columns={'baskv': 'kvj'})

    t2w.loc[:, ['geni', 'genj', 'ilj']] = 0
    t2w.loc[t2w['ibus'].isin(gens_buses), 'geni'] = 1
    t2w.loc[t2w['jbus'].isin(gens_buses), 'genj'] = 1
    t2w.loc[t2w['kvi'] <= t2w['kvj'], 'ilj'] = 1

    mask1 = (t2w['geni'] == 1) & (t2w['ilj'] == 1)
    keep1 = t2w.loc[mask1, 'jbus'].tolist()
    rem1 = t2w.loc[mask1, 'ibus'].tolist()
    mask2 = (t2w['genj'] == 1) & (t2w['ilj'] == 0)
    keep2 = t2w.loc[mask2, 'ibus'].tolist()
    rem2 = t2w.loc[mask2, 'jbus'].tolist()
    oldgenbuses=list(set(list(set(gens_buses) - set(rem1) - set(rem2)) + (keep1) + (keep2)))
    return gens_buses


def identify_PARs(trans2wdf, bus, all_branches_on, internals, PARVLIM):
    """
    Identify the Phase Angle Regulators (PARs).

    Args:
        trans2wdf (pandas.DataFrame): DataFrame containing transformer information.
        bus (pandas.DataFrame): DataFrame containing bus information.
        all_branches_on (pandas.DataFrame): DataFrame containing branches information.
        internals (list): List of internal areas.
        PARVLIM (float): PAR voltage limit.

    Returns:
        list: List of buses associated with PARs.
    """
    # Select PARs based on conditions
    PARs = trans2wdf.loc[
        (trans2wdf['cod1'].isin([-3, 3])) & (trans2wdf['stat']) == 1
    ]

    # use function add2lines_areaskv to add area and voltage level to PARs
    PARs_kV = utils.add2lines_businfo(
        PARs,
        bus,
        lefton1='ibus',
        righton1='ibus',
        lefton2='jbus',
        righton2='ibus',
    )

    # Filter PARs based on conditions
    PARs_selected = PARs_kV.loc[
        (PARs_kV['area_i'] == PARs_kV['area_j'])
        & (PARs_kV['area_i'].isin(internals))
        & (PARs_kV['baskv_i'] > PARVLIM)
    ]

    # Save retained PARs to file
    PARs_selected.drop(
        columns=['area_i', 'area_j', 'baskv_i', 'baskv_j']
    ).to_excel(
        '/home/alij/psse/InputData/REDUCTION/PARsRetained.xlsx', index=False
    )
    #
    # ret_lines_list = []
    # ret_lines_buses = []
    # for _, row in PARs_selected.iterrows():
    #     ibus = row['ibus']
    #     jbus = row['jbus']
    #
    #     con_ibus = all_branches_on.loc[
    #         (all_branches_on['ibus'] == ibus)
    #         | (all_branches_on['jbus'] == ibus),
    #         ['ibus', 'jbus', 'baskv_i', 'baskv_j', 'area_i', 'area_j'],
    #     ]
    #     con_ibus.loc[:, 'isline'] = 0
    #     con_ibus.loc[(con_ibus['baskv_i'] >= PARVLIM) &
    #                  (con_ibus['baskv_j'] >= PARVLIM), 'isline'] = 1
    #
    #     conj_bus = all_branches_on.loc[
    #         (all_branches_on['ibus'] == jbus)
    #         | (all_branches_on['jbus'] == jbus),
    #         ['ibus', 'jbus', 'baskv_i', 'baskv_j', 'area_i', 'area_j'],
    #     ]
    #     conj_bus.loc[:, 'isline'] = 0
    #     conj_bus.loc[
    #         (con_ibus['baskv_i'] >= PARVLIM) &
    #         (con_ibus['baskv_j'] >= PARVLIM), 'isline'] = 1
    #
    #     if (
    #         (con_ibus['isline'].sum() == con_ibus.shape[1])
    #         & (con_ibus['area_i'].isin(internals).sum() == con_ibus.shape[1])
    #         & (con_ibus['area_j'].isin(internals).sum() == con_ibus.shape[1])
    #     ):
    #
    #         ret_lines_list.append(row.reset_index().set_index('index').T)
    #         ret_lines_buses.append(
    #             list(
    #                 set(
    #                     [ibus, jbus]
    #                     + con_ibus['ibus'].tolist()
    #                     + con_ibus['jbus'].tolist()
    #                 )
    #             )
    #         )
    #     elif (
    #         (conj_bus['isline'].sum() == conj_bus.shape[1])
    #         & (conj_bus['area_i'].isin(internals).sum() == conj_bus.shape[1])
    #         & (conj_bus['area_j'].isin(internals).sum() == conj_bus.shape[1])
    #     ):
    #         ret_lines_list.append(row.reset_index().set_index('index').T)
    #         ret_lines_buses.append(
    #             list(
    #                 set(
    #                     [ibus, jbus]
    #                     + conj_bus['ibus'].tolist()
    #                     + conj_bus['jbus'].tolist()
    #                 )
    #             )
    #         )
    #     else:
    #         print('One line dropped')
    #
    # Get buses associated with retained PARs
    buses_to_retain = list(
        set(
            PARs_selected['ibus'].tolist()
            + PARs_selected['jbus'].tolist()
            + all_branches_on.loc[
                (all_branches_on['ibus'].isin(PARs_selected['ibus']))
                | (all_branches_on['jbus'].isin(PARs_selected['ibus'])),
                'ibus',
            ].tolist()
        )
    )
    return buses_to_retain


def get_border_buses(all_branches_on, buscapc, internals):
    """
    Get the border buses based on tie-line connections.

    Args:
        all_branches_on (pandas.DataFrame): DataFrame containing all branches information.
        buscapc (pandas.DataFrame): DataFrame containing bus capacity information.
        internals (list): List of internal areas.

    Returns:
        list: List of border buses.
    """
    ties = all_branches_on.loc[
        (
            (all_branches_on['area_i'].isin(internals))
            & ~(all_branches_on['area_j'].isin(internals))
        )
        | (
            (~all_branches_on['area_i'].isin(internals))
            & (all_branches_on['area_j'].isin(internals))
        )
    ]
    border_buses_i = ties.loc[ties['area_i'].isin(internals), 'ibus']
    border_buses_j = ties.loc[ties['area_j'].isin(internals), 'jbus']

    border_buses_tot = border_buses_i.tolist() + border_buses_j.tolist()

    border_buses_f = buscapc.loc[buscapc['ibus'].isin(border_buses_tot)]
    border_buses_final = border_buses_f.loc[
        (border_buses_f['count'] > 1)
        & (border_buses_f['baskv'] >= 30)
        & (border_buses_f['cap'] > 0)
    ]
    border_buses = border_buses_final['ibus'].tolist()
    return border_buses


def get_internal_retained_lines(
    all_branches_on, internals, kv_lim, rate_low_lim, rate_up_lim
):
    """
    Get the retained lines within internal areas.

    Args:
        all_branches_on (pandas.DataFrame): DataFrame containing all branches information.
        internals (list): List of internal areas.
        kv_lim (float): Voltage limit for retained lines.
        rate_low_lim (float): Lower rate limit for retained lines.
        rate_up_lim (float): Upper rate limit for retained lines.

    Returns:
        tuple: Tuple containing the list of retained buses and DataFrame of retained lines.
    """
    select_area = all_branches_on['area_i'].isin(internals)
    select_inside = all_branches_on['area_i'] == all_branches_on['area_j']
    select_lines = all_branches_on['baskv_i'] == all_branches_on['baskv_j']
    select_kvlim = all_branches_on['baskv_i'] > kv_lim
    select_ratellim = all_branches_on['rate1'] > rate_low_lim
    select_rateulim = all_branches_on['rate1'] < rate_up_lim
    select_posx = all_branches_on['x'] > 0
    select_longlines = all_branches_on['x'] > 0.0002
    select_nocable = all_branches_on['b'] < all_branches_on['x']
    select_minconi = all_branches_on['count_i'] > 2
    select_minconj = all_branches_on['count_j'] > 2
    select_aclines = all_branches_on['isac'] == 1

    int_ret_lines = all_branches_on.loc[
        select_lines
        & select_area
        & select_inside
        & select_kvlim
        & select_ratellim
        & select_rateulim
        & select_aclines
        & select_posx
        & select_nocable
        & select_longlines
        & select_minconi
        & select_minconj
    ]

    if int_ret_lines.empty:
        # Debugging
        return [], []

    ret_lines_list = []
    ret_lines_buses = []
    for _, row in int_ret_lines.iterrows():
        ibus = row['ibus']
        jbus = row['jbus']

        con_ibus = all_branches_on.loc[
            (all_branches_on['ibus'] == ibus)
            | (all_branches_on['jbus'] == ibus),
            ['ibus', 'jbus', 'baskv_i', 'baskv_j', 'area_i', 'area_j'],
        ]
        con_ibus.loc[:, 'isline'] = 0
        con_ibus.loc[con_ibus['baskv_i'] == con_ibus['baskv_j'], 'isline'] = 1

        conj_bus = all_branches_on.loc[
            (all_branches_on['ibus'] == jbus)
            | (all_branches_on['jbus'] == jbus),
            ['ibus', 'jbus', 'baskv_i', 'baskv_j', 'area_i', 'area_j'],
        ]
        conj_bus.loc[:, 'isline'] = 0
        conj_bus.loc[conj_bus['baskv_i'] == conj_bus['baskv_j'], 'isline'] = 1

        if (
            (con_ibus['isline'].sum() == con_ibus.shape[1])
            & (con_ibus['area_i'].isin(internals).sum() == con_ibus.shape[1])
            & (con_ibus['area_j'].isin(internals).sum() == con_ibus.shape[1])
        ):

            ret_lines_list.append(row.reset_index().set_index('index').T)
            ret_lines_buses.append(
                list(
                    set(
                        [ibus, jbus]
                        + con_ibus['ibus'].tolist()
                        + con_ibus['jbus'].tolist()
                    )
                )
            )
        elif (
            (conj_bus['isline'].sum() == conj_bus.shape[1])
            & (conj_bus['area_i'].isin(internals).sum() == conj_bus.shape[1])
            & (conj_bus['area_j'].isin(internals).sum() == conj_bus.shape[1])
        ):
            ret_lines_list.append(row.reset_index().set_index('index').T)
            ret_lines_buses.append(
                list(
                    set(
                        [ibus, jbus]
                        + conj_bus['ibus'].tolist()
                        + conj_bus['jbus'].tolist()
                    )
                )
            )
        else:
            print('One line dropped')

    # if list ret_lines_list not empty
    if ret_lines_list:
        retained_lines = pd.concat(ret_lines_list, ignore_index=True)
        retained_buses = [j for i in ret_lines_buses for j in i]
        # this is read by amapgenload.py file
        # XXX NOTE some of these lines are in parallel to each other
        retained_lines.to_csv('retainedlines_from_reduction.csv', index=False)
    else:
        retained_lines = pd.DataFrame()
        retained_buses = []
        print(' XXX No retained lines found XXX')

    return retained_buses, retained_lines


def add_iso(df, AREAS, areacol='area'):
    if 'ALLAREAS' in AREAS.keys():
        del AREAS['ALLAREAS']
    for i, (k, v) in enumerate(AREAS.items()):
        print(i, k, v)
        df.loc[df[areacol].isin(v), 'iso'] = k
    df['iso'] = df['iso'].fillna(value='ALLAREAS')
    return df
