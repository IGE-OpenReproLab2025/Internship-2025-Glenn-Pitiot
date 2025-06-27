# Author Sarah Vermaut 2025



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pandas import Series, DataFrame, Timedelta
from sklearn.linear_model import LinearRegression
from matplotlib.dates import DateFormatter, DayLocator, HourLocator



def read_csv_station(annee, station):
    """Read the data of meteo France station file and put it in a DataFrame

    Get the data and create a datetime index DataFrame

    """
    file = '~/private-storage/data_MF/'+annee+'_'+station+'_jour.csv'
    df = pd.read_csv(file, sep = '\t')
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    df.set_index('DATE', inplace=True)
    df = df.add_suffix('_'+station, axis=1)
    return df


def read_csv_lautaret(annee,fluxalpes_path) : 
    """Read the data from Flux'Alp station clean file

    Get the data and create a datetime index DataFrame

    """
    file = fluxalpes_path+annee+'_Lautaret_halfhour.csv'
    df = pd.read_csv(file, index_col=False)
    if annee in ['2022-2023', '2023-2024','2024-2025'] :
        df['datetime'] = pd.to_datetime(df['Unnamed: 0'])
        df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
        df.set_index('datetime', inplace=True)
    else :
        df['datetime'] = pd.to_datetime(df['TIMESTAMP'])
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df.set_index('datetime', inplace=True)
    return df


def make_no_snow_filter(albedo):
    """ Filter the grass from snow depth using albedo

    Filter created by dda, returns a serie where there should be no snow
    Create a smooth albedo to detect the no_snow period
    And then need to fill up the last time slot of the DataFrame
    
    """
   
    albedo_smooth = (albedo.resample('d').median()
                          .rolling(3,center=True,closed='both').median()
                          .rolling(3,center=True,closed='both').mean()
                          .bfill()
                     )
    
    albedo_smooth[albedo_smooth.index[-1]
                  +Timedelta('1D')-Timedelta('30min')] = albedo_smooth.iloc[-1]
    albedo = albedo_smooth.resample('30min').asfreq().interpolate(method='linear')
    
    no_snow = albedo<0.35
    return no_snow


def snow_accumulation(snow_depth) :
    """ Calculation of snow accumulation based on snow depth

    Returns the snow diff of a row with the other and the snow accumulation 
    
    """
    snow_diff = snow_depth.diff()
    snow_acc = snow_diff
    (snow_acc < 0) == 0
    return snow_diff, snow_acc


def densite_crocus(temp, vent):
    """ Calcul of freshly fallen snow density with crocus formula based on air temperature and wind speed

    """
    densite = 109 + 6*(temp) + 26*(vent**0.5)
    return densite


def catch_efficiency_unshield(temperature, wind) :
    """ Calcul catch efficieny for unshielded rain gauge

    Returns the effiency coefficient (between 0 and 1) based on the air temperature and wind speed
    Biblio : https://hess.copernicus.org/articles/21/3525/2017/ , Analysis of single-Alter-shielded and unshielded measurements of mixed and solid precipitation from WMO-SPICE, Kochendorfer (2017a) 

    """
    wind_speed = wind.copy()
    a = 0.0785
    b = 0.729
    c = 0.407
    for i, value in enumerate(wind_speed) :
        if value > 7.2 :
            wind_speed.iloc[i] = 7.2
    ce = np.exp(-(a*wind_speed)*(1-np.arctan(b*temperature)+c))
    return ce


def catch_efficiency_single_alter(temperature, wind) : 
    """ Calcul catch efficieny for a rain gauge with a single alter

    Returns the effiency coefficient (between 0 and 1) based on the air temperature and wind speed
    Biblio : https://hess.copernicus.org/articles/21/3525/2017/ , Analysis of single-Alter-shielded and unshielded measurements of mixed and solid precipitation from WMO-SPICE, Kochendorfer (2017a) 

    """
    wind_speed = wind.copy()
    a = 0.0348
    b = 1.366
    c = 0.779
    for i, value in enumerate(wind_speed) :
        if value > 7.2 :
            wind_speed.iloc[i] = 7.2
    ce = np.exp(-(a*wind_speed)*(1-np.arctan(b*temperature)+c))
    return ce


def convert_relativH_to_speH(relativ_H,temp_K, Patm_Pa) : 
    """Convert the relative humidity (%) into specific humidity (g/kg)

    Use the formulation of ???? to calculate the saturation vapour pressure. 
    Which is then used to calculate the water vapor pressure in the air, and with the atmospheric pressure come back to the specific pressure.

    """
    L_v = 2257000         # J/kg
    T3 = 273.16           # Triple point Temp.
    R_wat = 8.32/0.018    # Perfect gas constant/Molar mass of water vapour

    P_satvap = 611*np.exp(L_v*(1/T3 - 1/temp_K)/R_wat)   
    P_watvap = relativ_H/100*P_satvap
    speH = 0.622*P_watvap/(Patm_Pa-P_watvap)
    return speH

    
# Plot évolution des données précip, hauteur de neige et température +vent
def plot_temporal_ev_short_period(df):
    sns.set_theme()
    fig1,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    p1 = ax1.bar(df.index, df['Quantity_raw'], label = 'Precipitation (mm)', width = 0.02, color ='#1f77b4', edgecolor = 'none')
    p2 = ax2.plot(df.index, df['Snow_Depth'], label = 'Snow depth (m)', color = 'orange')
    ax1.set_ylabel('Cumulated rainfall (mm)',color ='#1f77b4')
    ax2.set_ylabel('Snow depth (m)', color='orange')
    ax1.set_ylim(0,4.25)
    ax2.set_ylim(0,2.12)
    ax1.tick_params('x', rotation =45)
    fig1.legend(bbox_to_anchor=(1.4,0.9))
    plt.title('Temporal evolution of precipitation and snow depth')


    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    ax1.plot(df.index, df['AirTC_Avg'], color ='#1f77b4')
    ax1.axhline(-1, color = 'black', linestyle='dashed')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_ylim(-10.2,5.5)
    ax2.plot(df.index, df['WindSpeed_Avg'], color ='#1f77b4')
    ax2.set_ylabel('Wind (m/s)')
    ax2.tick_params('x', rotation = 45)
    ax2.set_ylim(-0.2,10.5)


## Plot snow accumulation vs cumulated snow precipitation
def plot_acc_vs_cumul(df, var) : # periode par défaut '2018-2023', si plus court à préciser
    sns.set_theme()
    variable = {'Density' : 'density (kg/m3)', 'Snow_Depth_smooth' : 'snow depth (m)', 'Snow_Depth_acc' : 'snow accumulation (m)', 'AirTC_Avg' : "temperature (°C)", 'AirTC_Avg_mean' : "mean temperature (°C)",'AirTC_Avg_min' : "temperature min (°C)", 'AirTC_Avg_max' : "temperature max (°C)",  'Quantity_raw_snow' : 'snow precipitation (mm)', 'WindSpeed_Avg' : 'wind speed (m/s)', 'WindSpeed_Avg_min' : 'wind speed min (m/s)', 'WindSpeed_Avg_max' : 'wind speed max (m/s)',  'WindSpeed_Avg_mean' : 'mean wind speed (m/s)', 'HRair_Avg' : 'humidity (%)'}
    plt.figure()
    nom_variable =  str(variable.get(var))
    g = sns.scatterplot(data = df, x = 'Snow_Depth_acc', y = 'Quantity_raw_snow', hue = var)
    g.legend(title = nom_variable)
    g.set_xlabel('Snow accumulation (m)')
    g.set_ylabel('Cumulated solid rainfall (mm)')
    plt.title('Snow accumulation VS Cumulated snowfall depending on '+nom_variable)