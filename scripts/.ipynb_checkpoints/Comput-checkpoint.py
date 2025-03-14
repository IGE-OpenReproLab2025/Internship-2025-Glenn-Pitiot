#! /usr/bn/env python 

"""script.py : my first script"""
import pandas as pd

__author__  = "Glenn PITIOT"
__date__ = "14/03/2025"



def BM_permitivity(df, name_PA, name_PA_air_calibration=None):
    """
    Compute permittivity using basic electromagnetic laws.

    Parameters:
    df : DataFrame -> Data containing the necessary values.
    name_PA : str -> Name of the column in df referring to Time of Flight.
    name_PA_air_calibration : str (optional) -> Name of the column for air calibration (default: None).

    Returns:
    df : DataFrame -> Updated dataframe with a new column 'P_bm' (and 'P_bm_with_calib' if calibration is used).
    """
    
    # Constants
    L = 12e-2  # CS655 Length in meters
    c0 = 3e8   # Speed of light in vacuum (m/s)

    # Compute permittivity without calibration
    df['P_bm'] = (df[name_PA] * 1e-9 * c0 / (2 * L)) ** 2

    # If air calibration is provided, compute corrected permittivity
    if name_PA_air_calibration is not None:
        df['PA_1_calib'] = df[name_PA] - (df[name_PA_air_calibration] - 2 * L / c0 * 1e9)
        df['P_bm_with_calib'] = (df['PA_1_calib'] * 1e-9 * c0 / (2 * L)) ** 2
    
    return df