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
# import seaborn as sns
# from src.score import *

# HOW CAN YOU LOOK AT THE LOSSES OF THE CLIMATOLOGY PREDICTION
# AFTER THAT COMPARE IT TO THE PARAMETER LOSSES OF THE 10-7 UNET PREDICTION MODEL

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

norm_clm = transforms.Normalize((2.78415200e+02, -1.00402647e-01,  2.20140679e-01,  5.40906312e+04,
                                2.74440506e+02,  6.76697789e-01,  9.80986749e-05),
                                (2.11294838e+01, 5.57168569e+00, 4.77363485e+00, 3.35202722e+03,
                                1.55503555e+01, 3.62274453e-01, 3.57928990e-04))

# norm_clm = transforms.Normalize((2.78382787e+02, -8.24015036e-02,  2.22236745e-01,
#                                             5.40970992e+04, 2.74478000e+02,  6.73734047e-01,  9.93376168e-05),
#                                             (2.12668048e+01, 5.54037601e+00, 4.76958159e+00, 3.35185473e+03,
#                                             1.56055035e+01, 3.62924674e-01, 3.67389046e-04))


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

# whole_predictions = xr.open_mfdataset('/home/ge75tis/Desktop/oezyurt/climatology/yearly_pred.nc')
# data_whole = whole_predictions.to_array()
# np_whole = data_whole.to_numpy()
# np_whole_swap = np.swapaxes(np_whole, axis1=0, axis2=1)

torch_clm_week = torch.from_numpy(np_week_swap)
torch_clm_month = torch.from_numpy(np_month_swap)
# torch_clm_whole = torch.from_numpy(np_whole_swap)
norm_clm_week = norm_clm(torch_clm_week)
norm_clm_month = norm_clm(torch_clm_month)
# norm_clm_whole = norm_clm(torch_clm_whole)

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

# class CLMWHOLEDataset(Dataset):
#     def __init__(self):
#         self.norm_clm_whole = norm_clm_whole
#
#     def __getitem__(self, item):
#         return self.norm_clm_whole[item]
#
#     def __len__(self):
#         return len(norm_clm_whole)


val_dataset =  ValDataset()
val_loader = torch.utils.data.DataLoader(val_dataset)

clm_week_dataset = CLMWEEKDataset()
clm_week_loader = torch.utils.data.DataLoader(clm_week_dataset)

clm_month_dataset = CLMMONTHDataset()
clm_month_loader = torch.utils.data.DataLoader(clm_month_dataset)

# clm_whole_dataset = CLMWHOLEDataset()
# clm_whole_loader = torch.utils.data.DataLoader(clm_whole_dataset)

loss_type = nn.L1Loss()


clm_week_t2m = [0 for j in range(52)]
clm_week_u10 = [0 for j in range(52)]
clm_week_v10 = [0 for j in range(52)]
clm_week_z = [0 for j in range(52)]
clm_week_t = [0 for j in range(52)]
clm_week_tcc = [0 for j in range(52)]
clm_week_tp = [0 for j in range(52)]

clm_month_t2m = [0 for k in range(12)]
clm_month_u10 = [0 for k in range(12)]
clm_month_v10 = [0 for k in range(12)]
clm_month_z = [0 for k in range(12)]
clm_month_t = [0 for k in range(12)]
clm_month_tcc = [0 for k in range(12)]
clm_month_tp = [0 for k in range(12)]

clm_week_loss = [0 for j in range(52)]
clm_month_loss = [0 for k in range(12)]


index = 0
index_month = 0
t2m_loss = 0
u10_loss = 0
v10_loss = 0
z_loss = 0
t_loss = 0
tcc_loss = 0
tp_loss = 0
whole_loss = 0

week_whole = False
week_param = False
month_whole = False
month_param = False
ipp = 0

for i, (vals, clms) in enumerate(zip(val_loader, clm_week_loader)):
    vals = vals.to(device, dtype=torch.float32)
    pred = clms.to(device, dtype=torch.float32)

    t2m_loss += loss_type(pred.select(dim=1, index=0), vals.select(dim=1, index=0))
    u10_loss += loss_type(pred.select(dim=1, index=1), vals.select(dim=1, index=1))
    v10_loss += loss_type(pred.select(dim=1, index=2), vals.select(dim=1, index=2))
    z_loss += loss_type(pred.select(dim=1, index=3), vals.select(dim=1, index=3))
    t_loss += loss_type(pred.select(dim=1, index=4), vals.select(dim=1, index=4))
    tcc_loss += loss_type(pred.select(dim=1, index=5), vals.select(dim=1, index=5))
    tp_loss += loss_type(pred.select(dim=1, index=6), vals.select(dim=1, index=6))
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
        if(ipp == 31):
            index_month += 1
        if(ipp == 59):
            index_month += 1
        if(ipp == 90):
            index_month += 1
        if(ipp == 120):
            index_month += 1
        if(ipp == 151):
            index_month += 1
        if(ipp == 181):
            index_month += 1
        if(ipp == 212):
            index_month += 1
        if(ipp == 243):
            index_month += 1
        if(ipp == 273):
            index_month += 1
        if(ipp == 304):
            index_month += 1
        if(ipp == 334):
            index_month += 1
        if(ipp == 365):
            ipp = 0
            index_month = 0
        clm_month_loss[index_month] += loss_c.item()

    elif(week_param):
        if(i % 7 == 0 and i != 0):
            index += 1
            if (index == 52):
                index = 51

        if(i == 365):
            index = 0
        clm_week_t2m[index] += t2m_loss.item()
        clm_week_u10[index] += u10_loss.item()
        clm_week_v10[index] += v10_loss.item()
        clm_week_z[index] += z_loss.item()
        clm_week_t[index] += t_loss.item()
        clm_week_tcc[index] += tcc_loss.item()
        clm_week_tp[index] += tp_loss.item()

    elif(month_param):
        if(ipp == 31):
            index_month += 1
        if(ipp == 59):
            index_month += 1
        if(ipp == 90):
            index_month += 1
        if(ipp == 120):
            index_month += 1
        if(ipp == 151):
            index_month += 1
        if(ipp == 181):
            index_month += 1
        if(ipp == 212):
            index_month += 1
        if(ipp == 243):
            index_month += 1
        if(ipp == 273):
            index_month += 1
        if(ipp == 304):
            index_month += 1
        if(ipp == 334):
            index_month += 1
        if(ipp == 365):
            ipp = 0
            index_month = 0
        clm_month_t2m[index_month] += t2m_loss.item()
        clm_month_u10[index_month] += u10_loss.item()
        clm_month_v10[index_month] += v10_loss.item()
        clm_month_z[index_month] += z_loss.item()
        clm_month_t[index_month] += t_loss.item()
        clm_month_tcc[index_month] += tcc_loss.item()
        clm_month_tp[index_month] += tp_loss.item()

    ipp += 1
    # if(i % 7 == 0 and i != 0):
    #     index += 1
    #     if (index == 52):
    #         index = 51
    #
    # if(i % 730 == 0 and i != 0):
    #     index_month += 1
    #     if(index_month == 12):
    #         index_month = 11
    # #
    # if(i == 365):
    #     index = 0
    #     index_month = 0


whole_loss_per_param = [t2m_loss/len(val_dataset), u10_loss/len(val_dataset), v10_loss/len(val_dataset), z_loss/len(val_dataset),
                        t_loss/len(val_dataset), tcc_loss/len(val_dataset), tp_loss/len(val_dataset),]
print(whole_loss_per_param)



for i in range(52):
     if(i == 51):
        clm_week_t2m[i] = clm_week_t2m[i] / 16
        clm_week_u10[i] = clm_week_u10[i] / 16
        clm_week_v10[i] = clm_week_v10[i] / 16
        clm_week_z[i] = clm_week_z[i] / 16
        clm_week_t[i] = clm_week_t[i] / 16
        clm_week_tcc[i] = clm_week_tcc[i] / 16
        clm_week_tp[i] = clm_week_tp[i] / 16
        clm_week_loss[i] = clm_week_loss[i] / 16
     else:
        clm_week_t2m[i] = clm_week_t2m[i] / 14
        clm_week_u10[i] = clm_week_u10[i] / 14
        clm_week_v10[i] = clm_week_v10[i] / 14
        clm_week_z[i] = clm_week_z[i] / 14
        clm_week_t[i] = clm_week_t[i] / 14
        clm_week_tcc[i] = clm_week_tcc[i] / 14
        clm_week_tp[i] = clm_week_tp[i] / 14
        clm_week_loss[i] = clm_week_loss[i] / 14


for i in range(12):
    if(i == 0 or i == 2 or i == 4 or i == 6 or i == 7 or i == 9 or i == 11):
        clm_month_loss[i] = clm_month_loss[i] / 62
        clm_month_t2m[i] = clm_month_t2m[i] / 62
        clm_month_u10[i] = clm_month_u10[i] / 62
        clm_month_v10[i] = clm_month_v10[i] / 62
        clm_month_z[i] = clm_month_z[i] / 62
        clm_month_t[i] = clm_month_t[i] / 62
        clm_month_tcc[i] = clm_month_tcc[i] / 62
        clm_month_tp[i] = clm_month_tp[i] / 62
    elif(i == 3 or i == 5 or i == 8 or i == 10):
        clm_month_loss[i] = clm_month_loss[i] / 60
        clm_month_t2m[i] = clm_month_t2m[i] / 60
        clm_month_u10[i] = clm_month_u10[i] / 60
        clm_month_v10[i] = clm_month_v10[i] / 60
        clm_month_z[i] = clm_month_z[i] / 60
        clm_month_t[i] = clm_month_t[i] / 60
        clm_month_tcc[i] = clm_month_tcc[i] / 60
        clm_month_tp[i] = clm_month_tp[i] / 60
    elif(i == 1):
        clm_month_loss[i] = clm_month_loss[i] / 56
        clm_month_t2m[i] = clm_month_t2m[i] / 56
        clm_month_u10[i] = clm_month_u10[i] / 56
        clm_month_v10[i] = clm_month_v10[i] / 56
        clm_month_z[i] = clm_month_z[i] / 56
        clm_month_t[i] = clm_month_t[i] / 56
        clm_month_tcc[i] = clm_month_tcc[i] / 56
        clm_month_tp[i] = clm_month_tp[i] / 56


# # print(sum(clm_week_loss)/len(clm_week_loss))
# # print(clm_month_loss)
# # print(sum(clm_month_loss)/len(clm_month_loss))



fig = plt.figure()
if(week_param == True):
    plt.plot(clm_week_t2m, label = "t2m")
    plt.plot(clm_week_u10, label = "u10")
    plt.plot(clm_week_v10, label = "v10")
    plt.plot(clm_week_z, label = "z")
    plt.plot(clm_week_t, label = "t")
    plt.plot(clm_week_tcc, label = "tcc")
    plt.plot(clm_week_tp, label = "tp")
    plt.legend(loc='center right')
elif(week_whole == True):
    plt.plot(clm_week_loss)
elif(month_whole == True):
    plt.plot(clm_month_loss)
elif(month_param == True):
    plt.plot(clm_month_t2m, label = "t2m")
    plt.plot(clm_month_u10, label = "u10")
    plt.plot(clm_month_v10, label = "v10")
    plt.plot(clm_month_z, label = "z")
    plt.plot(clm_month_t, label = "t")
    plt.plot(clm_month_tcc, label = "tcc")
    plt.plot(clm_month_tp, label = "tp")
    plt.legend(loc='center right')


# fig.suptitle('climatology prediction L1 loss by week per parameter')
# plt.xlabel('weeks (every 7 days)')
# plt.ylabel('Average loss')
# plt.show()
# fig.savefig("/home/ge75tis/Desktop/per_param_clm_loss_by_week_L1")
