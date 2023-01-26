# from UNET import *
# import csv
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import zarr
# import xarray as xr
# import torch


## CLIMATOLOGY OLD PLOTS

# for i in range(52):
#      if(i == 51):
#         t2m_week_loss[i] = t2m_week_loss[i] / 16
#         u10_week_loss[i] = u10_week_loss[i] / 16
#         v10_week_loss[i] = v10_week_loss[i] / 16
#         z_week_loss[i] = z_week_loss[i] / 16
#         t_week_loss[i] = t_week_loss[i] / 16
#         tcc_week_loss[i] = tcc_week_loss[i] / 16
#         tp_week_loss[i] = tp_week_loss[i] / 16
#      else:
#         t2m_week_loss[i] = t2m_week_loss[i] / 14
#         u10_week_loss[i] = u10_week_loss[i] / 14
#         v10_week_loss[i] = v10_week_loss[i] / 14
#         z_week_loss[i] = z_week_loss[i] / 14
#         t_week_loss[i] = t_week_loss[i] / 14
#         tcc_week_loss[i] = tcc_week_loss[i] / 14
#         tp_week_loss[i] = tp_week_loss[i] / 14
#
# for i in range(12):
#     if(i == 0 or i == 2 or i == 4 or i == 6 or i == 7 or i == 9 or i == 11):
#         clm_month_t2m[i] = clm_month_t2m[i] / 62
#         clm_month_u10[i] = clm_month_u10[i] / 62
#         clm_month_v10[i] = clm_month_v10[i] / 62
#         clm_month_z[i] = clm_month_z[i] / 62
#         clm_month_t[i] = clm_month_t[i] / 62
#         clm_month_tcc[i] = clm_month_tcc[i] / 62
#         clm_month_tp[i] = clm_month_tp[i] / 62
#     elif(i == 3 or i == 5 or i == 8 or i == 10):
#         clm_month_t2m[i] = clm_month_t2m[i] / 60
#         clm_month_u10[i] = clm_month_u10[i] / 60
#         clm_month_v10[i] = clm_month_v10[i] / 60
#         clm_month_z[i] = clm_month_z[i] / 60
#         clm_month_t[i] = clm_month_t[i] / 60
#         clm_month_tcc[i] = clm_month_tcc[i] / 60
#         clm_month_tp[i] = clm_month_tp[i] / 60
#     elif(i == 1):
#         clm_month_t2m[i] = clm_month_t2m[i] / 56
#         clm_month_u10[i] = clm_month_u10[i] / 56
#         clm_month_v10[i] = clm_month_v10[i] / 56
#         clm_month_z[i] = clm_month_z[i] / 56
#         clm_month_t[i] = clm_month_t[i] / 56
#         clm_month_tcc[i] = clm_month_tcc[i] / 56
#         clm_month_tp[i] = clm_month_tp[i] / 56
#
# fig = plt.figure()
# plt.plot(clm_month_t2m, label="t2m")
# plt.plot(clm_month_u10, label="u10")
# plt.plot(clm_month_v10, label="v10")
# plt.plot(clm_month_z, label="z")
# plt.plot(clm_month_t, label="t")
# plt.plot(clm_month_tcc, label="tcc")
# plt.plot(clm_month_tp, label="tp")
# plt.legend(loc='center right')
#
# fig.suptitle('many-to-one model prediction L1 loss by month per parameter')
# plt.xlabel('calendar months')
# plt.ylabel('Average loss')
# plt.show()
# fig.savefig("/home/ge75tis/Desktop/per_param_unet_loss_by_month_L1")



## CONCATENATE ZARR DATASET YEARS AS ONE ZARR FILE

### new = wxee.xarray.DataArrayAccessor(xrdataset).normalize()
### test_array = new.to_numpy()
### norm_dataset = torch.from_numpy(test_array)

# years = list(range(2009, 2011))

# t2m = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/t2m/1979/t2m/')
# u10 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/u10/1979/u10/')
# v10 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/v10/1979/v10/')
# z = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/z/1979/z/')
# t = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/t/1979/t/')
# tcc = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tcc/1979/tcc/')
# tp = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tp/1979/tp/')
# print(len(t2m), len(u10), len(v10), len(z), len(t), len(tcc), len(tp))

# for x in years:
#     p = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TEST_DATA/tp/{x}/tp/'.format(x=x))
#     ###########tp.append(p)
#     print(len(tp))

# for x in years:
#     p = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/t2m/{x}/t2m/'.format(x=x))
#     p1 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/u10/{x}/u10/'.format(x=x))
#     p2 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/v10/{x}/v10/'.format(x=x))
#     p3 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/z/{x}/z/'.format(x=x))
#     p4 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/t/{x}/t/'.format(x=x))
#     p5 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tcc/{x}/tcc/'.format(x=x))
#     p6 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tp/{x}/tp/'.format(x=x))

# t2m = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/VALIDATION_DATA/t2m/2009/t2m/')
# u10 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/VALIDATION_DATA/u10/2009/u10/')
# v10 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/VALIDATION_DATA/v10/2009/v10/')
# z = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/VALIDATION_DATA/z/2009/z/')
# t = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/VALIDATION_DATA/t/2009/t/')
# tcc = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/VALIDATION_DATA/tcc/2009/tcc/')
# tp = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/VALIDATION_DATA/tp/2009/tp/')

# tcc = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/tcc/2009/tcc/')
# p = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/tcc/2010/tcc/')
#
# tcc.append(p)
# print(len(tcc))

# dataset = zarr.create((17520, 7, 32, 64))
# #
# dataset[:,0,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/t2m/2009/t2m/')
# dataset[:,1,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/u10/2009/u10/')
# dataset[:,2,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/v10/2009/v10/')
# dataset[:,3,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/z/2009/z/')
# dataset[:,4,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/t/2009/t/')
# dataset[:,5,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/tcc/2009/tcc/')
# dataset[:,6,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/tp/2009/tp/')
# # dataset[:,7,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/const_lsm/')
# # dataset[:,8,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/const_orog/')
# # dataset[:,9,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/const_slt/')
#
# print(dataset[17519])
# print(dataset[0])
#
# zarr.save('/home/ge75tis/Desktop/oezyurt/zarr dataset/concat/concatenated_val/', dataset)
#



## CONCAT CONSTANT VARIABLES LSM, OROG, SLT TO THE END OF EVERY DATASET

# lsm = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/const_lsm')
# orography = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/const_orog')
# slt = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/const_slt')
#
# print(lsm[0])
# print(orography[0])
# print(slt[0])

# slt_xr = xr.DataArray(slt)
# slt_np = slt_xr.to_numpy()
# nparray = np.empty((17520, 32, 64))
#
# for x in range(17520):
#     print(x)
#     nparray[x] = slt_np
#
# print(nparray[0])
# print(nparray[17519])
#
# zarr.save('/home/ge75tis/Desktop/oezyurt/zarr dataset/one_folder/validation_data_one_folder/const_slt/', nparray)


# # RESAMPLE TEST AND VALIDATION
#
# t2m = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TEST_DATA/t2m/2011/t2m/')
# u10 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TEST_DATA/u10/2011/u10/')
# v10 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TEST_DATA/v10/2011/v10/')
# z = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TEST_DATA/z/2011/z/')
# t = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TEST_DATA/t/2011/t/')
# tcc = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TEST_DATA/tcc/2011/tcc/')
# tp = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TEST_DATA/tp/2011/tp/')
#
# dataset = zarr.create((70128, 7, 32, 64))
#
# dataset = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/concat/const_concat_val/')
# print(dataset.shape)
#
# resampled_test = dataset[::24]
# print(resampled_test.shape)
# zarr.save('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_val', resampled_test)

