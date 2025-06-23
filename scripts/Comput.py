#! /usr/bn/env python 

"""comput.py : my first script"""

__author__  = "Glenn PITIOT"
__date__ = "23/06/2025/03/2025"

import pandas as pd
import numpy as np

def Ambach_Dent(Perm,D_bulk):
    '''Compute liquid water content using Ambacht et Dent (1980)
    
    𝑊𝑣 = (𝜀 - 1 – 1.202 * Ds – 0.983 * Ds²)/21.3  
    
    Permitivity : List dialectric permittivity 
    Density_bulk : List bulk density [kg/m^3]
    
    return
    lwc : liquid water content 

    documentation can be find in SCRIPT NOTEBOOKS / AMBACH
    '''
    Perm = np.array(Perm)
    D_bulk = np.array(D_bulk)
    Lwc = D_bulk*0
    for i in range(5):
        Ds = D_bulk/1000 - Lwc
        Lwc = (Perm- 1 - 1.202*Ds - 0.983*Ds**2)/21.3
    return Lwc

def Comput_LWC(df_LWC_profile,df_density,column_name_density : str ,column_name_permitivity : str):
    ''' Comput permittivity from a value of density and a permittivity, Use of Ambach et Dent, 
    df_LWC_profile : data frame with permittivity profile
    df_density : data frame with density profile
    column_name_density : name of density column
    column_name_permitivity :name of permittivity column

    '''
    ########################################################################################################
    # Convert permittivy with density into Liquid Water content. 
    ########################################################################################################
    
    # First step we need to merge closer values of permitivity and density respect to elveation. ###########
    
    # Ajouter un ID temporaire basé sur l’index initial
    df_LWC_profile['_order'] = df_LWC_profile.index  
    
    # Trier les DataFrames avant la fusion
    df_LWC_profile = df_LWC_profile.sort_values('Z')
    df_density = df_density.sort_values('Z')
    
    # Fusion avec la bonne direction
    df_LWC_profile = pd.merge_asof(df_LWC_profile, df_density, on='Z', direction='nearest')
    
    # Restaurer l’ordre d’origine et réassigner les ID correctement
    df_LWC_profile = df_LWC_profile.sort_values('_order')  # Tri pour remettre dans l’ordre initial
    df_LWC_profile.index = range(len(df_LWC_profile))  # Réaffecte des ID dans l’ordre correct
    df_LWC_profile = df_LWC_profile.drop(columns=['_order'])  # Nettoyage

    # Comput Liquid water content with Ambach and Dent law##################################################
    
    df_LWC_profile['LWC [vol%]'] = Ambach_Dent(df_LWC_profile[column_name_permitivity], df_LWC_profile[column_name_density])

    # All variable under 0 are put to 0. There are on the invald zone of Ambatch function.

    df_LWC_profile.loc[df_LWC_profile['LWC [vol%]']<0,'LWC [vol%]'] = 0
    
    return(df_LWC_profile)

def comput_density(w_sc : float, df ,column_name: str)->pd.DataFrame:
    '''
    Input :
    w_sc : Empty weight of the snow cutter [kg]
    df : Data frame of density 
    colmun name : name of the column for Cutter weight

    Comput only valide for snow cutter from IGE:
    d = (w_f - w_sc)*4
    
    Ouptut: 
    df_density : data frame with a new column 'density [kg/m3]'
    '''
    # Conversion into density: density = weight / volume with
    df['density [kg/m3]'] = (df[column_name] - w_sc*1000) * 4
    df = df.drop(columns=['Cutter_weight_[g]'])
    return(df)
    


