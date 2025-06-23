#! /usr/bn/env python 

"""comput.py : my first script"""

__author__  = "Glenn PITIOT"
__date__ = "16/04/2025"

import pandas as pd
import numpy as np
import re

def creation_profile_final(df_density,df_LWC_profile,df_SSA_profile):
    """
    Creation of a profile from 3 data frame: Density, LWC, SSA profile
    df_grid is the finnest data frame which is use as the reference grid.
    """
    # df_density Use of density profil because he as the fiinest size,
    
    df_final = df_density
    
    # Merge LWC table respect to the nereast Z . #############################################################################
    
    # Ajouter un ID temporaire basé sur l’index initial

    df_final['_order'] = df_final.index  
    
    # Trier les DataFrames avant la fusion

    df_final = df_final.sort_values('Z')
    df_LWC_profile = df_LWC_profile.sort_values('Z')
    # Fusion avec la bonne direction
    df_final = pd.merge_asof(df_final, df_LWC_profile, on='Z', direction='nearest')
    # Restaurer l’ordre d’origine et réassigner les ID correctement
    df_final = df_final.sort_values('_order')  # Tri pour remettre dans l’ordre initial
    df_final.index = range(len(df_final))  # Réaffecte des ID dans l’ordre correct
    df_final = df_final.drop(columns=['_order','density [kg/m3]_y'])  # Nettoyage
    
    # Merge SSA table respect to the nereast Z . #############################################################################
    
    # Ajouter un ID temporaire basé sur l’index initial
    df_final['_order'] = df_final.index  
    
    # Trier les DataFrames avant la fusion
    df_final = df_final.sort_values('Z')
    df_SSA_profile = df_SSA_profile.sort_values('Z')
    
    # Fusion avec la bonne direction
    df_final = pd.merge_asof(df_final, df_SSA_profile, on='Z', direction='nearest')
    
    # Restaurer l’ordre d’origine et réassigner les ID correctement
    df_final = df_final.sort_values('_order')  # Tri pour remettre dans l’ordre initial
    df_final.index = range(len(df_final))  # Réaffecte des ID dans l’ordre correct
    df_final = df_final.drop(columns=['_order'])  # Nettoyage
    
    
    
    df_final = df_final.rename(columns={'density [kg/m3]_x' : 'density [kg/m3]' })
    return(df_final)

def create_profile(df_density, df_LWC_profile, df_SSA_profile):
    """
    Create a unified vertical profile by merging Density, LWC, and SSA dataframes.

    The function automatically selects the dataframe with the finest vertical resolution (smallest median spacing in Z)
    and uses it as the reference grid. The other dataframes are merged onto this grid using nearest-neighbor matching
    along the Z-axis (vertical coordinate).

    Parameters:
    - df_density (pd.DataFrame): DataFrame containing density values and a 'Z' column.
    - df_LWC_profile (pd.DataFrame): DataFrame containing LWC values and a 'Z' column.
    - df_SSA_profile (pd.DataFrame): DataFrame containing SSA values and a 'Z' column.

    Returns:
    - pd.DataFrame: Merged DataFrame on the finest Z-grid with all available parameters.
    """
    
    import numpy as np

    # Dtype forcing
    df_LWC_profile['Z'] = df_LWC_profile['Z'].astype('float64')
    df_density['Z'] = df_density['Z'].astype('float64')

    # Compute median vertical resolution (Z spacing) for each dataframe
    res_density = df_density['Z'].sort_values().diff().median()
    res_lwc     = df_LWC_profile['Z'].sort_values().diff().median()
    res_ssa     = df_SSA_profile['Z'].sort_values().diff().median()

    # Associate each DataFrame with its resolution and name
    df_list = [
        ('density', res_density, df_density),
        ('lwc',     res_lwc,     df_LWC_profile),
        ('ssa',     res_ssa,     df_SSA_profile),
    ]

    # Select the DataFrame with the finest vertical resolution (smallest median Z step)
    df_list_sorted = sorted(df_list, key=lambda x: x[1])
    grid_name, _, df_grid = df_list_sorted[0]
    df_final = df_grid.copy()

    # Helper function to merge another profile onto the reference grid using nearest Z
    def merge_on_Z(df_base, df_to_merge):
        df_base['_order'] = df_base.index  # Save original order
        df_base = df_base.sort_values('Z')
        df_to_merge = df_to_merge.sort_values('Z')

        df_merged = pd.merge_asof(df_base, df_to_merge, on='Z', direction='nearest')
        df_merged = df_merged.sort_values('_order')
        df_merged.index = range(len(df_merged))
        return df_merged.drop(columns=['_order'])

    # Merge the two non-reference profiles onto the reference grid
    for name, _, df_other in df_list:
        if name != grid_name:
            df_final = merge_on_Z(df_final, df_other)

    # Clean up duplicated or unnecessary columns (especially density)
    if 'density [kg/m3]_x' in df_final.columns:
        df_final = df_final.rename(columns={'density [kg/m3]_x': 'density [kg/m3]'})
        if 'density [kg/m3]_y' in df_final.columns:
            df_final = df_final.drop(columns=['density [kg/m3]_y'])

    return df_final

    


import numpy as np

import numpy as np

def manip_prep_Thickness_Snow(prep):
    """
    Computes an estimate of the total thickness of the snowpack (in meters)
    based on weighted snow (WSN_VEG) and real snow (RSN_VEG) values 
    across multiple vegetation layers in an xarray Dataset.

    The thickness is calculated by summing the ratio of WSN_VEG to RSN_VEG
    for each available layer (from 1 to 49), while ignoring any negative values.
    Invalid divisions (e.g., divide-by-zero) are handled by replacing them with 0.

    Parameters:
    -----------
    prep : xarray.Dataset
        Dataset containing snow-related variables named in the pattern:
        'WSN_VEG1', 'RSN_VEG1', ..., 'WSN_VEG49', 'RSN_VEG49'.

    Returns:
    --------
    None
        Prints the estimated snowpack thickness in meters.
    """
    
    H = 0  # Initialize total snowpack thickness

    for i in range(1, 50):
        wsn_col = f'WSN_VEG{i}'
        rsn_col = f'RSN_VEG{i}'

        if wsn_col in prep.data_vars and rsn_col in prep.data_vars:
            wsn_values = prep[wsn_col].values
            rsn_values = prep[rsn_col].values

            # Mask negative values in both arrays
            valid_mask = (wsn_values >= 0) & (rsn_values > 0)  # rsn must be > 0 to avoid div by 0
            wsn_clean = np.where(valid_mask, wsn_values, 0)
            rsn_clean = np.where(valid_mask, rsn_values, 1)  # Set denominator to 1 where invalid, to keep ratio = 0

            # Safe division with NumPy
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.divide(wsn_clean, rsn_clean)
                ratio = np.where(np.isfinite(ratio), ratio, 0)

            H += np.sum(ratio)  # Sum of the current layer’s valid ratios

    print("Thickness snow pack:", H, "m")




def manip_prep_netcdf2dataframe(prep):
    """
    Extracts specific vegetation-related variables from a NetCDF xarray Dataset
    and organizes them into a structured pandas DataFrame.

    Parameters:
    -----------
    prep : xarray.Dataset
        An xarray Dataset containing variables with names following the pattern
        <BASE><INDEX>, where BASE is one of a set of target variable names and
        INDEX is a layer number.

    Returns:
    --------
    pd.DataFrame
        A DataFrame indexed by 'layer' (couche) containing extracted values for each
        of the target variables, with NaNs where data is missing.
    """

    # List of target variable families to extract
    target_variables = ["RSN_VEG", "WSN_VEG", "HSN_VEG", "SAG_VEG", "SG1_VEG", "SG2_VEG", "SHI_VEG"]

    # Regular expression pattern to extract base name and index from variable names
    pattern = re.compile(r"([A-Z0-9_]+?)(\d+)$")

    # Dictionary to store values for each variable and layer index
    var_dict = {var: {} for var in target_variables}

    # Loop through each variable in the Dataset
    for var_name in prep.data_vars:
        match = pattern.match(var_name)
        if match:
            base, index = match.groups()
            index = int(index)

            if base in target_variables:
                val = prep[var_name].values
                # Extract a representative value (scalar or first element)
                if val.ndim == 0:
                    var_dict[base][index] = val.item()
                else:
                    var_dict[base][index] = val.flatten()[0]

    # Determine the maximum number of layers to include in the DataFrame
    max_layer = max((max(d.keys()) for d in var_dict.values() if d), default=-1) + 1
    layers = list(range(max_layer))
    
    # Build the DataFrame with NaNs for missing values
    prep_vis = pd.DataFrame(index=layers)
    for var in target_variables:
        values = [var_dict[var].get(i, np.nan) for i in layers]
        prep_vis[var] = values

    prep_vis.index.name = "layer"
    return prep_vis
