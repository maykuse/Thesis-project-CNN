import torch.nn.init
import torchvision.transforms
from UNET import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import zarr
import xarray as xr
import matplotlib.pyplot as plt
import pandas


# t2m = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/t2m_train/*.nc', combine='by_coords')
# u10 = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/u10_train/*.nc', combine='by_coords')
# v10 = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/v10_train/*.nc', combine='by_coords')
# z = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/z_train/*.nc', combine='by_coords')
# t = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/t_train/*.nc', combine='by_coords').drop('level')
# tcc = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/tcc_train/*.nc', combine='by_coords')
# tp = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/train_nc/tp_train/*.nc', combine='by_coords')
#
#
# t2m_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/t2m/*.nc', combine='by_coords')
# u10_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/u10/*.nc', combine='by_coords')
# v10_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/v10/*.nc', combine='by_coords')
# z_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/z/*.nc', combine='by_coords')
# t_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/t/*.nc', combine='by_coords').drop('level')
# tcc_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/tcc/*.nc', combine='by_coords')
# tp_val = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/nc dataset/val_nc/tp/*.nc', combine='by_coords')
#
#
# train_data = xr.merge([t2m, u10, v10, z, t, tcc, tp])
# valid_data = xr.merge([t2m_val, u10_val, v10_val, z_val, t_val, tcc_val, tp_val])
#
#
# train_data = train_data.isel(time=slice(None, None, 24))
# valid_data = valid_data.isel(time=slice(None, None, 24))
#
#
# def create_weekly_climatology_forecast(ds_train, valid_time):
#     ds_train['week'] = ds_train['time.week']
#     weekly_averages = ds_train.groupby('week').mean('time')
#     valid_time['week'] = valid_time['time.week']
#     fc_list = []
#     for tt in valid_time:
#         fc_list.append(weekly_averages.sel(week=tt.week))
#     return xr.concat(fc_list, dim=valid_time)
#
# def create_monthly_climatology_forecast(ds_train, valid_time):
#     ds_train['month'] = ds_train['time.month']
#     monthly_averages = ds_train.groupby('month').mean('time')
#     valid_time['month'] = valid_time['time.month']
#     fc_list = []
#     for tt in valid_time:
#         fc_list.append(monthly_averages.sel(month=tt.month))
#     return xr.concat(fc_list, dim=valid_time)
#
# def create_yearly_climatology_forecast(ds_train, valid_time):
#     whole_average = ds_train.mean('time')
#     fc_list = []
#     for t in valid_time:
#         fc_list.append(whole_average)
#     return xr.concat(fc_list, dim=valid_time)
#

# climatology = create_monthly_climatology_forecast(train_data, valid_data.time)
# print(climatology)
# climatology.to_netcdf('/home/ge75tis/Desktop/oezyurt/climatology/true_monthly_pred.nc')
# print('finished')





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

norm = transforms.Normalize((2.78415200e+02, -1.00402647e-01,  2.20140679e-01,  5.40906312e+04,
                                2.74440506e+02,  6.76697789e-01,  9.80986749e-05,  3.37078289e-01,
                                3.79497583e+02,  6.79204298e-01),
                            (2.11294838e+01, 5.57168569e+00, 4.77363485e+00, 3.35202722e+03,
                            1.55503555e+01, 3.62274453e-01, 3.57928990e-04, 4.59003773e-01,
                            8.59872249e+02, 1.16888408e+00))
# This is the mean and std of the resampled training data (not the climatology prediction)
norm_clm = transforms.Normalize((2.78415200e+02, -1.00402647e-01,  2.20140679e-01,  5.40906312e+04,
                                2.74440506e+02,  6.76697789e-01,  9.80986749e-05),
                                (2.11294838e+01, 5.57168569e+00, 4.77363485e+00, 3.35202722e+03,
                                1.55503555e+01, 3.62274453e-01, 3.57928990e-04))


validation_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_val')

xr_val = xr.DataArray(validation_set)
np_val = xr_val.to_numpy()
torch_val = torch.from_numpy(np_val)
norm_val = norm(torch_val)

monthly_predictions = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/climatology/true_monthly_pred.nc')
data_month = monthly_predictions.to_array()
np_month = data_month.to_numpy()
np_month_swap = np.swapaxes(np_month, axis1=0, axis2=1)

weekly_predictions = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/climatology/true_weekly_pred.nc')
data_week = weekly_predictions.to_array()
np_week = data_week.to_numpy()
np_week_swap = np.swapaxes(np_week, axis1=0, axis2=1)


torch_clm_week = torch.from_numpy(np_week_swap)
torch_clm_month = torch.from_numpy(np_month_swap)
norm_clm_week = norm_clm(torch_clm_week)
norm_clm_month = norm_clm(torch_clm_month)

class ValDataset(Dataset):
    def __init__(self):
        self.norm_val = norm_val
    def __getitem__(self, item):
        return self.norm_val[item]
    def __len__(self):
        return len(norm_val)


class CLMWEEKDataset(Dataset):
    def __init__(self):
        self.norm_clm_week = norm_clm_week

    def __getitem__(self, item):
        return self.norm_clm_week[item]

    def __len__(self):
        return len(norm_clm_week)


class CLMMONTHDataset(Dataset):
    def __init__(self):
        self.norm_clm_month = norm_clm_month

    def __getitem__(self, item):
        return self.norm_clm_month[item]

    def __len__(self):
        return len(norm_clm_month)



val_dataset =  ValDataset()
val_loader = torch.utils.data.DataLoader(val_dataset)

clm_week_dataset = CLMWEEKDataset()
clm_week_loader = torch.utils.data.DataLoader(clm_week_dataset)

clm_month_dataset = CLMMONTHDataset()
clm_month_loader = torch.utils.data.DataLoader(clm_month_dataset)


loss_type = nn.L1Loss()


clm_week_param = [[0 for i in range(52)] for j in range(7)]
clm_month_param = [[0 for i in range(12)] for j in range(7)]

clm_week_loss = [0 for j in range(52)]
clm_month_loss = [0 for k in range(12)]
num_of_days_monthly = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
inc_point = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


index = 0
index_month = 0
whole_loss = 0
param_loss = [0 for i in range(7)]

week_whole = False
week_param = False
month_whole = True
month_param = False
counter = 0

for i, (vals, clms) in enumerate(zip(val_loader, clm_month_loader)):
    vals = vals.to(device, dtype=torch.float32)
    pred = clms.to(device, dtype=torch.float32)

    for j in range(7):
        param_loss[j] = loss_type(pred.select(dim=1, index=j), vals.select(dim=1, index=j))

    loss_c = loss_type(pred, vals[:, :7])

    if(week_whole):
        if(i % 7 == 0 and i != 0):
            index += 1
            if (index == 52):
                index = 51

        if(i == 365):
            index = 0
        clm_week_loss[index] += loss_c.item()

    elif(month_whole):
        if(counter in inc_point):
            index_month += 1
        if(counter == 365):
            counter = 0
            index_month = 0
        clm_month_loss[index_month] += loss_c.item()

    elif(week_param):
        if(i % 7 == 0 and i != 0):
            index += 1
            if (index == 52):
                index = 51

        if(i == 365):
            index = 0
        for j in range(7):
            clm_week_param[j][index] += param_loss[j].item()

    elif(month_param):
        if (counter in inc_point):
            index_month += 1
        if(counter == 365):
            counter = 0
            index_month = 0
        for j in range(7):
            clm_month_param[j][index] += param_loss[j].item()

    counter += 1

for i in range(52):
     if(i == 51):
        for j in range(7):
            clm_week_param[j][i] = clm_week_param[j][i] / 16

        clm_week_loss[i] = clm_week_loss[i] / 16
     else:
        for j in range(7):
            clm_week_param[j][i] = clm_week_param[j][i] / 14
        clm_week_loss[i] = clm_week_loss[i] / 14


for i in range(12):
    if(i == 0 or i == 2 or i == 4 or i == 6 or i == 7 or i == 9 or i == 11):
        for j in range(7):
            clm_month_param[j][i] = clm_month_param[j][i] / 62
        clm_month_loss[i] = clm_month_loss[i] / 62

    elif(i == 3 or i == 5 or i == 8 or i == 10):
        for j in range(7):
            clm_month_param[j][i] = clm_month_param[j][i] / 60
        clm_month_loss[i] = clm_month_loss[i] / 60
    elif(i == 1):
        for j in range(7):
            clm_month_param[j][i] = clm_month_param[j][i] / 56
        clm_month_loss[i] = clm_month_loss[i] / 56

print('Week losses per param')
for j in range(7):
    print(clm_week_param[j])
print('-----------')
print('Month Losses per param')
for j in range(7):
    print(clm_month_param[j])

print("Whole losses")
print(clm_week_loss)
print(clm_month_loss)


labels = ["t2m", "u10", "v10", 'z', 't', "tcc", "tp"]
fig = plt.figure()
if(week_param):
    for i in range(7):
        plt.plot(clm_week_param[i], label = labels[i])
    plt.legend(loc='center right')
elif(week_whole):
    plt.plot(clm_week_loss)
elif(month_whole):
    plt.plot(clm_month_loss)
elif(month_param):
    for i in range(7):
        plt.plot(clm_month_param[i], label = labels[i])
    plt.legend(loc='center right')


# fig.suptitle('climatology prediction L1 loss by week per parameter')
# plt.xlabel('weeks (every 7 days)')
# plt.ylabel('Average loss')
# plt.show()
# fig.savefig("/home/ge75tis/Desktop/per_param_clm_loss_by_week_L1")
