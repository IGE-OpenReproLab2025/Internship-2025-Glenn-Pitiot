
#! /usr/bn/env python 

""""""

__author__  = "Glenn PITIOT"
__date__ = "27/06/2025/"

import pandas as pd
import file_functions as fct # Crocus formula


def calculate_precipitation_smoothed(df_sacc, interval:int, threshold:float)->pd.DataFrame:
    """
    Calculate and smooth snowfall precipitation over time intervals,
    considering only events with snow accumulation greater than a threshold.

    Parameters:
        df_sacc (pd.DataFrame): DataFrame with 'Snow_Depth', 'AirTC_Avg', and 'WindSpeed_Avg'.
        interval (int): Number of rows between start and end of interval.
        threshold (float): Minimum snow accumulation (in meters) to consider as a snowfall event.

    Returns:
        pd.DataFrame: Original DataFrame with 'Precip_Calculated' column filled in.
    """
    df_sacc['Precip_Calculated'] = 0.

    for i in range(len(df_sacc) - interval):
        df_start = df_sacc.iloc[[i]]
        df_end = df_sacc.iloc[[i + interval]]

        # Calculate snow accumulation
        snow_acc = df_end.Snow_Depth.values - df_start.Snow_Depth.values

        # Only process events with significant accumulation
        if snow_acc > threshold:
            T_mean = (df_start.AirTC_Avg.values + df_end.AirTC_Avg.values) / 2
            WindSpeed_mean = (df_start.WindSpeed_Avg.values + df_end.WindSpeed_Avg.values) / 2

            # Compute density using CROCUS model
            density_crocus = fct.densite_crocus(T_mean, WindSpeed_mean)
            precip_total = density_crocus * snow_acc

            # Distribute precipitation evenly over the interval
            precip_per_step = precip_total.item() / interval
            interval_indices = df_sacc.index[i : i + interval]

            df_sacc.loc[interval_indices, 'Precip_Calculated'] = precip_per_step

    return df_sacc

