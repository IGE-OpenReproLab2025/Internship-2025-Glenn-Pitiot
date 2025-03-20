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
    #df['P_bm'] = (df[name_PA] / df[name_PA_air_calibration])
    df['P_bm'] = (df[name_PA] * 1e-9 * c0 / (2 * L)) ** 2
    
    # If air calibration is provided, compute corrected permittivity
    if name_PA_air_calibration is not None:
        df['PA_1_calib'] = df[name_PA] - (df[name_PA_air_calibration] - 2 * L / c0 * 1e9)
        df['P_bm_with_calib'] = (df['PA_1_calib'] * 1e-9 * c0 / (2 * L)) ** 2
    
    return df


def generate_step_values_for_density_plot(x_points, y_points,thickness):
    '''
    Processed data for generation of density plot

    Input
    x_point : List of mesures beging at the top oh the snow layer 
    y_point : List of elevation for each values of mesures begining at the top

    Output:
    x_values, yvalues for plot of step
    '''
    
    x_values = [x_points[0],x_points[0]]  # On commence à x = 0
    y_values = [thickness,y_points[0]]  # Première valeur de y

    for i in range(len(x_points)):
        x_values.extend([x_points[i], x_points[i]])
        y_values.extend([y_points[i], y_points[i] if i < len(y_points) - 1 else None])

    x_values.pop()  # Supprime le dernier élément inutile de x_values
    y_values.pop()  # Supprime le dernier élément inutile de y_values

    return x_values, y_values

def Ambach_Dent(Perm,D_bulk):
    '''From Ambacht et Dent (1980)
    Permitivity : List dialectric permittivity 
    Density_bulk : List bulk density [kg/m^3]
    return
    lwc : liquid water content 
    '''
    lwc = []
    for i in range(10):
        Ds = D_bulk/1000 - lwc
        lwc = (Perm- 1 - 1.202*Ds - 0.983*Ds**2)/21.3
    return lwc

