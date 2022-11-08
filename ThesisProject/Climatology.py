import numpy as np
import xarray as xr
import zarr
import matplotlib.pyplot as plt
# import seaborn as sns
# from src.score import *

# HOW CAN YOU LOOK AT THE LOSSES OF THE CLIMATOLOGY PREDICTION
# AFTER THAT COMPARE IT TO THE PARAMETER LOSSES OF THE 10-7 UNET PREDICTION MODEL

t2m = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/t2m_train/*.nc', combine='by_coords')
u10 = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/u10_train/*.nc', combine='by_coords')
v10 = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/v10_train/*.nc', combine='by_coords')
z = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/z_train/*.nc', combine='by_coords')
t = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/t_train/*.nc', combine='by_coords').drop('level')
tcc = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/tcc_train/*.nc', combine='by_coords')
tp = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/tp_train/*.nc', combine='by_coords')


t2m_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/t2m/*.nc', combine='by_coords')
u10_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/u10/*.nc', combine='by_coords')
v10_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/v10/*.nc', combine='by_coords')
z_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/z/*.nc', combine='by_coords')
t_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/t/*.nc', combine='by_coords').drop('level')
tcc_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/tcc/*.nc', combine='by_coords')
tp_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/tp/*.nc', combine='by_coords')


train_data = xr.merge([t2m, u10, v10, z, t, tcc, tp])
valid_data = xr.merge([t2m_val, u10_val, v10_val, z_val, t_val, tcc_val, tp_val])


def create_weekly_climatology_forecast(ds_train, valid_time):
    ds_train['week'] = ds_train['time.week']
    weekly_averages = ds_train.groupby('week').mean('time')
    valid_time['week'] = valid_time['time.week']
    fc_list = []
    for tt in valid_time:
        fc_list.append(weekly_averages.sel(week=tt.week))
    return xr.concat(fc_list, dim=valid_time)


# weekly_climatology = create_weekly_climatology_forecast(train_data, valid_data.time)
# print(weekly_climatology)
# weekly_climatology.to_netcdf('/home/ge75tis/Desktop/oezyurt/climatology/2nd_weekly_pred.nc')
# print('finished')
# weekly_climatology.tp.plot()