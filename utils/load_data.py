#! /usr/bn/env python 

"""comput.py : my first script"""

__author__  = "Glenn PITIOT"
__date__ = "16/04/2025"

import pandas as pd
import numpy as np
import re

def load_TDR_data(absolute_path,name_path):
    '''
    Load TDR data from CSV, remove first data abnd convert data into numeric
    '''
    # Creat a data frame with all measures.
    df_TDR = pd.read_csv(absolute_path+name_path, delimiter=',',header=1,low_memory=False)
    # Remove 1 ligne which contain units and also first mesurment which corespond to the open snow pit ect 
    df_TDR = df_TDR.drop(range(150))
    
    # Date convertion
    df_TDR["TIMESTAMP"] = pd.to_datetime(df_TDR["TIMESTAMP"], format="%Y-%m-%d %H:%M:%S")
    
    # Convert all columns except 'TIMESTAMP' to numeric
    cols_to_convert = df_TDR.columns.difference(["TIMESTAMP"])  # Select all columns except 'TIMESTAMP'
    df_TDR[cols_to_convert] = df_TDR[cols_to_convert].apply(pd.to_numeric, errors="coerce")
    return df_TDR


def load_PT100_data(absolute_path,name_path,date_seuil):
    '''
    Load of PT100_data with a seuil date
    '''
    # Creat a data frame with all measures.
    df_PT100 = pd.read_csv(absolute_path+name_path, delimiter=',',header=1,low_memory=False)
    
    # Remove 1 ligne which contain units
    df_PT100 = df_PT100.drop([0,1])
    
    # Date convertion
    df_PT100["TIMESTAMP"] = pd.to_datetime(df_PT100["TIMESTAMP"], format="%Y-%m-%d %H:%M:%S")
    
    # Convert all columns except 'TIMESTAMP' to numeric
    cols_to_convert = df_PT100.columns.difference(["TIMESTAMP"])  # Select all columns except 'TIMESTAMP'
    df_PT100[cols_to_convert] = df_PT100[cols_to_convert].apply(pd.to_numeric, errors="coerce")
    
    
    # Date de filtrage
    date_seuil = pd.to_datetime(date_seuil)
    
    # Filtrage
    return( df_PT100[df_PT100['TIMESTAMP'] > date_seuil])

def load_concatenate_data(data_folder=str,Date=list):
    '''
    Load of all snow pit refere to the Date
    Date is list of date as : Date = ['18-03-2025','21-03-2025','2025-03-31_12h44','2025-03-31_15h18']
    '''
    Dic_df_density={}
    file_name = ".csv"
    for date in Date:
        Dic_df_density[date] =  pd.read_csv(data_folder+'concatenate-data_cdl_'+date+".csv", delimiter=',')
    return Dic_df_density
