# Data processing libs.
import numpy as np
import xarray as xr

# Visualisation libs.
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Set parameters.
# Parent coordinate file.
pcf = xr.open_dataset('/mnt/localssd/Data_nemo/Meshes_domains/Coordinates/Global/ORCA_R36_coord_new.nc').squeeze()
x_middle = int(pcf['x'].size / 2)

# Enter Pacific and Atlantic y-indicies (latitude-like).
pac_first_yind = 7550
pac_last_yind = -2
atl_first_yind = 7350
atl_last_yind = -1

# Enter first and last x-indicies (latitude-like). It must be less than 1/2 x-dimention size!
pac_first_xind = 1500
pac_last_xind = 5900
atl_first_xind = pac_last_xind + (x_middle - pac_last_xind) * 2 + 1
atl_last_xind = pcf['x'].size - pac_first_xind + 1

# Saving properties
target_path = '/mnt/localssd/Data_nemo/Meshes_domains/Coordinates/Regional'
target_name = 'arct_cutorca36_coord.nc'

# Patch processing
def grid_selector(pcf, var, extent, pac_patch=False):
    '''
    Functon that select and cut 2D arrays from parent global ORCA coordinate file.

    Parameters
    ----------
    pcf : xarray Dataset
        Parent global ORCA Coordinate File.
    var : str
        Grid variable name from pcf.
    extent : list
        List with indicies to cut [y_min, y_max, x_min, x_max].
    atl_patch : bool
        Pacific (True) and Atlantic (False) switch (default is False)

    Returns
    -------
    grid_array
        Ndarray to put in patch dataset.
    '''

    # grid type lists
    t_vars = ['nav_lon', 'nav_lat', 'glamt', 'gphit', 'e1t', 'e2t']
    u_vars = ['glamu', 'gphiu', 'e1u', 'e2u']
    v_vars = ['glamv', 'gphiv', 'e1v', 'e2v']
    f_vars = ['glamf', 'gphif', 'e1f', 'e2f']

    if var not in pcf:
        raise ValueError(f"There is no variable named {var} in the parent coordinate file.")

    if pac_patch:  # Pacific patch selection
        if var in t_vars:
            grid_array = np.flip(pcf[var].sel(y=slice(extent[0], extent[1]), x=slice(extent[2], extent[3])).values)
        elif var in u_vars:
            grid_array = np.flip(pcf[var].sel(y=slice(extent[0], extent[1]), x=slice(extent[2] - 1, extent[3] - 1)).values)
        elif var in v_vars:
            grid_array = np.flip(pcf[var].sel(y=slice(extent[0] - 1, extent[1] - 1), x=slice(extent[2], extent[3])).values)
        elif var in f_vars:
            grid_array = np.flip(pcf[var].sel(y=slice(extent[0] - 1, extent[1] - 1), x=slice(extent[2] - 1, extent[3] - 1)).values)
    else:  # Atlantic patch selection
        grid_array = pcf[var].sel(y=slice(extent[0], extent[1]), x=slice(extent[2], extent[3])).values

    return grid_array

# Dataset creation
def create_dataset(pcf, extent, pac_patch=False):
    '''
    Function to create a dataset for a given extent and patch type.

    Parameters
    ----------
    pcf : xarray Dataset
        Parent global ORCA Coordinate File.
    extent : list
        List with indices to cut [y_min, y_max, x_min, x_max].
    pac_patch : bool
        Pacific (True) and Atlantic (False) switch (default is False).

    Returns
    -------
    dataset
        xarray Dataset for the given extent.
    '''
    return xr.Dataset(
        data_vars=dict(
            nav_lon=(["y", "x"], grid_selector(pcf, 'nav_lon', extent, pac_patch)),
            nav_lat=(["y", "x"], grid_selector(pcf, 'nav_lat', extent, pac_patch)),
            glamt=(["y", "x"], grid_selector(pcf, 'glamt', extent, pac_patch)),
            glamu=(["y", "x"], grid_selector(pcf, 'glamu', extent, pac_patch)),
            glamv=(["y", "x"], grid_selector(pcf, 'glamv', extent, pac_patch)),
            glamf=(["y", "x"], grid_selector(pcf, 'glamf', extent, pac_patch)),
            gphit=(["y", "x"], grid_selector(pcf, 'gphit', extent, pac_patch)),
            gphiu=(["y", "x"], grid_selector(pcf, 'gphiu', extent, pac_patch)),
            gphiv=(["y", "x"], grid_selector(pcf, 'gphiv', extent, pac_patch)),
            gphif=(["y", "x"], grid_selector(pcf, 'gphif', extent, pac_patch)),
            e1t=(["y", "x"], grid_selector(pcf, 'e1t', extent, pac_patch)),
            e1u=(["y", "x"], grid_selector(pcf, 'e1u', extent, pac_patch)),
            e1v=(["y", "x"], grid_selector(pcf, 'e1v', extent, pac_patch)),
            e1f=(["y", "x"], grid_selector(pcf, 'e1f', extent, pac_patch)),
            e2t=(["y", "x"], grid_selector(pcf, 'e2t', extent, pac_patch)),
            e2u=(["y", "x"], grid_selector(pcf, 'e2u', extent, pac_patch)),
            e2v=(["y", "x"], grid_selector(pcf, 'e2v', extent, pac_patch)),
            e2f=(["y", "x"], grid_selector(pcf, 'e2f', extent, pac_patch))
        )
    )

# Atlantic patch as xarray Dataset
atl_extent = [atl_first_yind, atl_last_yind, atl_first_xind, atl_last_xind]
atl_dataset = create_dataset(pcf, atl_extent)

# Pacific patch as xarray Dataset
pac_extent = [pac_first_yind, pac_last_yind, pac_first_xind, pac_last_xind]
pac_dataset = create_dataset(pcf, pac_extent, pac_patch=True)

# Combine datasets and save
whole_dataset = xr.concat([atl_dataset, pac_dataset], dim='y')
whole_dataset.to_netcdf(f'{target_path}/{target_name}')

# Visualization
def plot_dataset(dataset, title):
    '''
    Function to visualize the dataset.

    Parameters
    ----------
    dataset : xarray Dataset
        Dataset to visualize.
    title : str
        Title of the plot.
    '''
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    dataset['nav_lon'].plot(ax=ax, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title(title)
    plt.show()

# Plot Atlantic and Pacific patches
plot_dataset(atl_dataset, 'Atlantic Patch')
plot_dataset(pac_dataset, 'Pacific Patch')