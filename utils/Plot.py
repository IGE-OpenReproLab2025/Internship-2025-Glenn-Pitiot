

__author__  = "Glenn PITIOT"
__date__ = "03/04/2025"

import matplotlib.pyplot as plt # used for makings nice plots
import pandas as pd
import numpy as np


def generate_step_values_for_profile_plot(x_points, y_points, thickness) -> tuple[list[float], list[float]]:
    '''
    Generates stepwise x and y values for a centered step plot of a snow profile.

    Inputs:
    x_points : List of measured values (e.g., density), from top to bottom
    y_points : List of elevation/depths for each value, starting from the top
    thickness : Total height of the snowpack profile

    Outputs:
    x_values, y_values : Step plot coordinates for visualization
    '''

    x_values = []
    y_values = []

    # Calculate midpoints between each pair of y values for transitions
    mid_y = []
    for i in range(len(y_points) - 1):
        mid_y.append((y_points[i] + y_points[i + 1]) / 2)

    # Start from the top of the snowpack
    y_values.append(thickness)
    x_values.append(x_points[0])

    # Build step values centered between measurement points
    for i in range(len(mid_y)):
        y_values.extend([mid_y[i], mid_y[i]])
        x_values.extend([x_points[i], x_points[i + 1]])

    return x_values, y_values


def plot_multiple_days(Date,Dic_df,legend=True):
    '''
    plot Density LWC and SSA for the selected Date
    '''
    fig, axs = plt.subplots(1,3,figsize=(2*3.54,1.25*3.54),sharey=True)
    print(axs)
    for date in Date:
        
        x,y = generate_step_values_for_profile_plot(Dic_df[date]['density [kg/m3]'],Dic_df[date].Z*10e-3,Dic_df[date]['Z'][0]*1e-2)
        axs[0].step(x,y,where='post', label='Day :'+date)
    
        x,y = generate_step_values_for_profile_plot(Dic_df[date]['LWC [vol%]']*100,Dic_df[date].Z*10e-3,Dic_df[date]['Z'][0]*1e-2)
        axs[1].step(x,y,where='post', label='Day :'+date)
    
        x,y = generate_step_values_for_profile_plot(Dic_df[date]['SSA_avg'],Dic_df[date].Z*10e-3,Dic_df[date]['Z'][0]*1e-2)
        axs[2].step(x,y,where='post', label='Day :'+date)
    
    # Labels and titles
    axs[0].grid(True)
    axs[0].set_xlabel('Density [$kg/m^3$]')
    axs[0].set_ylabel('Snow Depth [$m$]')
    
    axs[1].grid(True)
    axs[1].set_xlabel('LWC [%]')   
    
    axs[2].grid(True)
    axs[2].set_xlabel('SSA [$m^2/kg$]')
    if legend:
        plt.legend()
    plt.tight_layout()

    return fig, axs
def plot_strati_single_day(df, layers_cm, delta_surface=0.0):
    '''
    Plot Density, LWC, and SSA profiles with stratigraphy bands as background on each subplot,
    and add a legend for grain types.

    Parameters:
    - df: pandas DataFrame containing columns 'Z', 'density [kg/m3]', 'LWC [vol%]', and 'SSA_avg'
    - layers_cm: list of tuples (start_cm, end_cm, grain_type)
    - delta_surface: float, offset (in meters) to align stratigraphy to profile surface

    Notes:
    - Depth values are in cm in the data and layers, but plotted in meters.
    '''

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    color_map = {
        "small grains": 'c',
        "big grains": 'royalblue',
        "Crust": 'grey',
        "Alternation big/small grains": 'plum',
        "nan": 'white',
        "Other": 'white'
    }

    fig, axs = plt.subplots(1, 3, figsize=(9, 6), sharey=True)

    # Convert stratigraphy layers to meters and apply vertical offset
    layers_m = [(start / 100 - delta_surface, end / 100 - delta_surface, grain_type)
                for start, end, grain_type in layers_cm]

    for ax in axs:
        # Background stratigraphy
        for start_m, end_m, grain_type in layers_m:
            color = color_map.get(grain_type, 'white')
            ax.axhspan(start_m, end_m, facecolor=color, alpha=0.3, edgecolor='none', zorder=0)

    # Convert Z to meters
    Z_m = df['Z'] / 100.0

    # Plot Density
    axs[0].step(df['density [kg/m3]'], Z_m, where='post', label='Density', zorder=5)
    axs[0].set_xlabel('Density [kg/m3]')
    axs[0].set_title('Density')

    # Plot LWC
    axs[1].step(df['LWC [vol%]'] * 100, Z_m, where='post', label='LWC', zorder=5)
    axs[1].set_xlabel('LWC [%]')
    axs[1].set_title('LWC')

    # Plot SSA
    axs[2].step(df['SSA_avg'], Z_m, where='post', label='SSA', zorder=5)
    axs[2].set_xlabel('SSA [m²/kg]')
    axs[2].set_title('SSA')

    # Formatting
    for ax in axs:
        ax.grid(True)

    axs[0].set_ylabel('Depth [m]')

    # Legend for stratigraphy grain types
    handles = [mpatches.Patch(color=c, label=l, alpha=0.3) for l, c in color_map.items() if c != 'white']
    axs[2].legend(handles=handles, loc='upper center', bbox_to_anchor=(1, -0.15),
                  fontsize=9, ncol=2, frameon=False)

    plt.tight_layout()
    plt.show()
    return fig, axs


def plot_TDR(df_TDR,TDR, Dic_TDR, var_name = str, date = tuple):
    '''
    plt of all TDR 
    '''
    # Date de filtrage
    (date_debut,date_fin) = date
    date_debut = pd.to_datetime(date_debut)
    date_fin = pd.to_datetime(date_fin)
    
    # Filtrage
    df_TDR = df_TDR[df_TDR['TIMESTAMP'] > date_debut]
    df_TDR = df_TDR[df_TDR['TIMESTAMP'] < date_fin]
    
    
    
    plt.figure(figsize=(10,5))
    for tdr in TDR:
        plt.plot(df_TDR['TIMESTAMP'],df_TDR[var_name+'_{}_Avg'.format(tdr)],label=Dic_TDR[tdr])
    Label ={'P' : 'Permitivity', 'PA':'Time of flight [ns]','VR': 'Voltage Ratio', 'Tt': 'Temperature [°C]'}
    plt.xlabel('Date')
    plt.ylabel(Label[var_name])
    plt.legend()
    plt.title("TDR Time of flight")
    plt.xticks(rotation=90)
    plt.grid()

def plot_PT100(df_PT100,PT100, Dic_PT100, date =tuple):
    ''' 
    plot for PT100 using PT100 number of PT100 and Dic for label 
    '''
    
    # Date de filtrage
    (date_debut,date_fin) = date
    date_debut = pd.to_datetime(date_debut)
    date_fin = pd.to_datetime(date_fin)
    
    # Filtrage
    df_PT100 = df_PT100[df_PT100['TIMESTAMP'] > date_debut]
    df_PT100 = df_PT100[df_PT100['TIMESTAMP'] < date_fin]

    df_PT100=df_PT100.sort_values('TIMESTAMP')
    plt.figure(figsize=(10,5))
    for pt100 in PT100:
        plt.plot(df_PT100['TIMESTAMP'],df_PT100['T({})'.format(pt100)],label=Dic_PT100[pt100])
    plt.xlabel('Date')
    plt.ylabel('Temperature [°C]')
    plt.legend()
    plt.title("PT100 Température")
    plt.xticks(rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

    
def plot_interpolated_heatmap_PT100(df_PT100,Dic_PT100, date=tuple, resolution=0.01, vmin=None, vmax=None, cmap='jet', ax=None):
    '''
    Plot an interpolated heatmap of PT100 temperatures over time and height.

    Parameters:
    - df_PT100: DataFrame with 'TIMESTAMP' column and 'T(n)' columns for PT100 readings
    - Dic_PT100: Dictionary {n: 'height in cm'}, mapping PT100 numbers to height
    - date: Tuple of (start_date, end_date) as strings
    - resolution: Vertical resolution for interpolation in meters (e.g., 0.01 for 1 cm)
    - vmin: Minimum temperature for the color scale (optional)
    - vmax: Maximum temperature for the color scale (optional)
    - cmap: Colormap (default: 'jet')
    - ax: Optional matplotlib axes object to plot on
    '''

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    import matplotlib.dates as mdates
    
    # Convert date strings to datetime
    start_date, end_date = pd.to_datetime(date[0]), pd.to_datetime(date[1])

    # Filter dataframe by date
    df_filtered = df_PT100[(df_PT100['TIMESTAMP'] >= start_date) & (df_PT100['TIMESTAMP'] <= end_date)].copy()

    # Sort PT100s by physical height
    PT100_sorted = sorted(Dic_PT100.keys(), key=lambda x: float(Dic_PT100[x].replace(' cm','')))
    heights_cm = [float(Dic_PT100[pt].replace(' cm', '')) for pt in PT100_sorted]
    heights_m = np.array(heights_cm) / 100  # Convert to meters

    # Prepare time and interpolated height axis
    time = pd.to_datetime(df_filtered['TIMESTAMP']).values
    height_interp = np.arange(min(heights_m), max(heights_m), resolution)

    # Initialize interpolated temperature matrix
    temp_interp_matrix = []

    for i in range(len(df_filtered)):
        temps = [df_filtered[f'T({pt})'].iloc[i] for pt in PT100_sorted]
        f_interp = interp1d(heights_m, temps, kind='linear', fill_value='extrapolate')
        temp_interp_matrix.append(f_interp(height_interp))

    temp_interp_matrix = np.array(temp_interp_matrix).T  # shape: (height, time)

    # Create meshgrid
    time_grid, height_grid = np.meshgrid(time, height_interp)

    # Use provided axis or create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    # Plotting
    contour = ax.contourf(
        time_grid,
        height_grid,
        temp_interp_matrix,
        levels=100,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    ax.set_xlim(time[0], time[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

    # Colorbar (only if creating figure)
    if ax is None:
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Temperature [°C]')
        cmap.set_over((0.32, 0.0, 0.097))
        cbar.set_label('Temperature [°C]')
        ax.set_title('Interpolated Temperature Profile Over Time and Height')
    else:
        cbar = plt.colorbar(contour, ax=ax)

    #plt.tight_layout()

    return ax  # return ax in case further customization is needed

def plot_density_profile(df_density, thickness):
    """
    plot of density profile
    """
    plt.figure(figsize=(2.34,2*2.34 ))
    x,y = generate_step_values_for_profile_plot(df_density['density [kg/m3]'],df_density.Z*10e-3,thickness)
    plt.step(x,y,where='post',label ='df_density_9h30')
    plt.xlabel('Density [kg/m^3]')
    plt.ylabel('z [m]')
    plt.title("Density profile")
    plt.legend()
    plt.grid()

def plot_LWC(df_LWC_profile,thickness):
    """
    Plot of Liquid water content profile
    """
    plt.figure(figsize=(2.34,2*2.34 ))
    x,y = generate_step_values_for_profile_plot(df_LWC_profile['LWC [vol%]'],df_LWC_profile.Z*10e-3,thickness)
    plt.step(np.array(x)*100,y,where='post',label='df_LWC_profile_13h00')
    # Labels and titles
    plt.xlabel('LWC [vol.%]')
    plt.ylabel('z [m]')
    plt.title("LWC profile")
    plt.legend()
    plt.grid()