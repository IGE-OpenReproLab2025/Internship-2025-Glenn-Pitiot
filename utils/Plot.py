

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

    
def plot_interpolated_heatmap_PT100(df_PT100, Dic_PT100, date=tuple, resolution=0.01, vmin=None, vmax=None, cmap='jet', ax=None, cbar=True):
    """
    Plot an interpolated heatmap of temperature profiles from PT100 sensors over time.

    Parameters
    ----------
    df_PT100 : pandas.DataFrame
        DataFrame containing temperature measurements from PT100 sensors. 
        Must include a 'TIMESTAMP' column and columns named as 'T(sensor_name)' 
        corresponding to the keys in `Dic_PT100`.

    Dic_PT100 : dict
        Dictionary mapping each PT100 sensor name (e.g., 'PT1') to its physical height 
        as a string in centimeters (e.g., '10 cm').

    date : tuple of str or pandas.Timestamp
        A tuple specifying the start and end date for the plot (e.g., ('2025-03-01', '2025-03-17')).

    resolution : float, optional (default=0.01)
        Vertical resolution of the interpolation in meters.

    vmin : float, optional
        Minimum temperature value for color normalization (used for clipping the colormap).

    vmax : float, optional
        Maximum temperature value for color normalization (used for clipping the colormap).

    cmap : str, optional (default='jet')
        Colormap used for the heatmap.

    ax : matplotlib.axes.Axes, optional
        Existing matplotlib Axes object to plot on. If None, a new figure and axes are created.

    cbar : bool, optional (default=True)
        Whether to display the colorbar.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.

    contour : QuadContourSet
        The contour set created by `ax.contourf`, useful for further customization.

    Notes
    -----
    - Temperatures are interpolated linearly along the vertical axis using `scipy.interpolate.interp1d`.
    - Time is converted to matplotlib's internal ordinal format for plotting.
    - The function clips temperature values to [vmin, vmax] if both are provided.

    Example
    -------
    >>> ax, contour = plot_interpolated_heatmap_PT100(df_PT100, Dic_PT100, date=('2025-03-01', '2025-03-17'))
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    import matplotlib.dates as mdates
    from matplotlib.colors import Normalize

    # Convert date strings to datetime objects
    start_date, end_date = pd.to_datetime(date[0]), pd.to_datetime(date[1])

    # Filter the DataFrame by the selected date range
    df_filtered = df_PT100[(df_PT100['TIMESTAMP'] >= start_date) & (df_PT100['TIMESTAMP'] <= end_date)].copy()

    # Sort PT100 sensors by their physical height
    PT100_sorted = sorted(Dic_PT100.keys(), key=lambda x: float(Dic_PT100[x].replace(' cm','')))
    heights_cm = [float(Dic_PT100[pt].replace(' cm', '')) for pt in PT100_sorted]
    heights_m = np.array(heights_cm) / 100  # Convert from centimeters to meters

    # Prepare the time axis in ordinal format (matplotlib uses float days since 0001-01-01)
    time_dt = pd.to_datetime(df_filtered['TIMESTAMP'])
    time_ord = mdates.date2num(time_dt)

    # Create the interpolated vertical axis
    height_interp = np.arange(min(heights_m), max(heights_m), resolution)

    # Perform vertical interpolation for each timestamp
    temp_interp_matrix = []
    for i in range(len(df_filtered)):
        temps = [df_filtered[f'T({pt})'].iloc[i] for pt in PT100_sorted]
        f_interp = interp1d(heights_m, temps, kind='linear', fill_value='extrapolate')
        temp_interp_matrix.append(f_interp(height_interp))

    temp_interp_matrix = np.array(temp_interp_matrix).T  # shape: (height, time)

    # Clip data between vmin and vmax if both are defined
    if vmin is not None and vmax is not None:
        temp_interp_matrix = np.clip(temp_interp_matrix, vmin, vmax)
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    else:
        norm = None

    # Create meshgrid for plotting (time x height)
    time_grid, height_grid = np.meshgrid(time_ord, height_interp)

    # Use the provided axis or create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    # Plot filled contours using interpolated temperature data
    contour = ax.contourf(
        time_grid,
        height_grid,
        temp_interp_matrix,
        levels=100,
        cmap=cmap,
        norm=norm  # Use clipped normalization if provided
    )

    ax.set_xlim(time_ord[0], time_ord[-1])
    ax.xaxis_date()  # Interpret x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))  # Format tick labels as 'day-month'

    # Add colorbar if requested
    if cbar:
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Temperature [°C]')

    return ax, contour

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
    plt.grid

def select_date(date, time):
    """
    Selects a boolean mask for elements in 'time' that fall between the provided start and end dates.

    Parameters:
    - date: tuple or list of two date strings or datetime objects (start, end)
    - time: array-like of datetime values (e.g., pd.Series, list, np.array)

    Returns:
    - List of booleans (mask): True if time[i] is within the date interval
    """
    start, end = pd.to_datetime(date[0]), pd.to_datetime(date[1])
    mask = [(start < t < end) for t in pd.to_datetime(time)]
    return mask

from snowtools.plots.stratiprofile.profilPlot import saisonProfil

def plot_obs_vs_BKT_RCH(dz_BKT,temp_BKT,time_BKT,
                        dz_RCH,temp_RCH,time_RCH,
                        df_PT100,Dic_PT100,
                        date,
                        vmin,vmax):
    """
    Plot temperature profiles from BKT and RCH sensors alongside interpolated PT100 observations.

    This function creates a figure with three vertically stacked subplots sharing the y-axis (snow height).
    The first two subplots display temperature profiles from BKT and RCH datasets respectively,
    while the third shows a heatmap of interpolated PT100 temperature measurements over time and height.

    Parameters
    ----------
    dz_BKT : array-like
        Snow height values for BKT measurements (in meters).
    temp_BKT : array-like
        Temperature values for BKT measurements (in Kelvin).
    time_BKT : array-like of datetime-like
        Timestamps corresponding to BKT measurements.

    dz_RCH : array-like
        Snow height values for RCH measurements (in meters).
    temp_RCH : array-like
        Temperature values for RCH measurements (in Kelvin).
    time_RCH : array-like of datetime-like
        Timestamps corresponding to RCH measurements.

    df_PT100 : pandas.DataFrame
        DataFrame containing PT100 temperature observations, including a 'TIMESTAMP' column.
    Dic_PT100 : dict
        Dictionary mapping PT100 sensor names to their heights (strings with units, e.g. '50 cm').

    date : tuple of str or datetime-like
        Start and end dates to filter the data for plotting (e.g. ('2025-03-01', '2025-03-31')).

    vmin : float
        Minimum temperature value for color scaling.
    vmax : float
        Maximum temperature value for color scaling.

    Returns
    -------
    matplotlib.figure.Figure, numpy.ndarray, matplotlib.contour.QuadContourSet
        The matplotlib Figure object, array of Axes objects, and the contour set of the PT100 heatmap.
    """
    # Appliquer les masques
    masque_BKT = select_date(date, time_BKT)
    masque_RCH = select_date(date, time_RCH)
    
    fig, ax = plt.subplots(3, 1, figsize=(2*3.54, 1.5*3.54), sharey=True,constrained_layout=True)
    
    saisonProfil(
                ax[0],
                dz_BKT[masque_BKT],
                temp_BKT[masque_BKT] - 273.16,
                time_BKT[masque_BKT],
                colormap="Blues",
                value_min=vmin,
                value_max=vmax,
                legend="Temperature [$°C$]",
                cbar_show=False
    )
    
    #ax[0].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=3))
    
    print(ax[0].get_xticks())
    saisonProfil(
                ax[1],
                dz_RCH[masque_RCH],
                temp_RCH[masque_RCH] - 273.16,
                time_RCH[masque_RCH],
                colormap="Blues",
                value_min=vmin,
                value_max=vmax,
                legend="Temperature [$°C$]",
                ylimit=1.42,
                cbar_show=False
    )
    
    ax[2],contour = plot_interpolated_heatmap_PT100(
                                                    df_PT100=df_PT100,
                                                    Dic_PT100=Dic_PT100,
                                                    date=('2025-03-17 15:00','2025-03-29 15:00'),
                                                    resolution=0.01,
                                                    vmin=vmin,
                                                    vmax=vmax,
                                                    cmap="Blues",
                                                    ax=ax[2],
                                                    cbar=False
    )
    for a in ax:
        a.set_ylabel("Snow Height [$m$]")
    
    ax[2].set_xlabel("Date")
    
    ax[0].plot([], [], label='(BKT)')
    ax[1].plot([], [], label='(RCH)')
    ax[2].plot([], [], label='(OBS)')
    
    # Appeler legend sans titre
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
    cbar.set_label('Temperature [°C]')

import matplotlib.dates as mdates

def plot_BKTvsRCH(dz_BKT,density_BKT,snowliq_BKT,time_BKT,
                  dz_RCH,density_RCH,snowliq_RCH,time_RCH,
                  ):
    """
    Plot density and liquid water content (LWC) profiles from BKT and RCH sensors
    in a 2x2 grid of subplots for comparison.

    Uses the `saisonProfil` plotting function from snowtools to visualize profiles
    over time and snow height with appropriate color maps.

    Parameters
    ----------
    dz_BKT : array-like
        Snow height values for BKT measurements (in meters).
    density_BKT : array-like
        Density values for BKT measurements (in kg/m³).
    time_BKT : array-like of datetime-like
        Timestamps corresponding to BKT measurements.

    dz_RCH : array-like
        Snow height values for RCH measurements (in meters).
    density_RCH : array-like
        Density values for RCH measurements (in kg/m³).
    time_RCH : array-like of datetime-like
        Timestamps corresponding to RCH measurements.

    Returns
    -------
    None
        Displays the plots but does not return any object.
    """
    from snowtools.plots.stratiprofile.profilPlot import saisonProfil
    
    #ax = plt.gca()
    fig, ax = plt.subplots(2,2,figsize=(2.5*3.54,1.2*3.54), sharey=True, sharex=True)
    
    
    rect1 = saisonProfil(ax[0,0], dz_BKT, density_BKT, time_BKT, colormap='Purples',legend="Density [$kg/m^3$]")
    cbar = saisonProfil(ax[1,0], dz_RCH, density_RCH, time_RCH, colormap='Purples',legend="Density [$kg/m^3$]")
    
    rect1 = saisonProfil(ax[0,1], dz_BKT, snowliq_BKT, time_BKT, colormap='Blues',legend="LWC [$kg/m^3$]")
    cbar = saisonProfil(ax[1,1], dz_RCH, snowliq_RCH, time_RCH, colormap='Blues',legend="LWC [$kg/m^3$]")
    
    ax[0,0].set_ylabel("Snow Height [$m$]")
    ax[1,0].set_ylabel("Snow Height [$m$]")
    
    
    # Date formating
    date_format = mdates.DateFormatter('%d-%b')
    
    ax[0,0].plot([], [], label=' ')  # a space as label
    ax[1,0].plot([], [], label=' ')
    ax[0,1].plot([], [], label=' ')
    ax[1,1].plot([], [], label=' ')
    
    ax[0,0].legend(title='(BKT)', loc='upper right')
    ax[1,0].legend(title='(RCH)', loc='upper right')
    ax[0,1].legend(title='(BKT)', loc='upper right')
    ax[1,1].legend(title='(RCH)', loc='upper right')
    
    plt.show()