from UNET import *
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
import zarr

# COPIED FROM TRAINING TESTING START
### new = wxee.xarray.DataArrayAccessor(xrdataset).normalize()
### test_array = new.to_numpy()
### norm_dataset = torch.from_numpy(test_array)

# dataset = zarr.create((262985, 7, 32, 64))
# dataset[:,0,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/t2m/1979/t2m/')
# dataset[:,1,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/u10/1979/u10/')
# dataset[:,2,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/v10/1979/v10/')
# dataset[:,3,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/z/1979/z/')
# dataset[:,4,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/t/1979/t/')
# dataset[:,5,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tcc/1979/tcc/')
# dataset[:,6,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tp/1979/tp/')


# years = list(range(1980, 2009))
#
# t2m = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/t2m/1979/t2m/')
# u10 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/u10/1979/u10/')
# v10 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/v10/1979/v10/')
# z = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/z/1979/z/')
# t = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/t/1979/t/')
# tcc = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tcc/1979/tcc/')
# tp = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tp/1979/tp/')
# print(len(t2m), len(u10), len(v10), len(z), len(t), len(tcc), len(tp))

# for x in years:
#     p = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tp/{x}/tp/'.format(x=x))
#     ###########tp.append(p)
#     print(len(tp))
#
# # for x in years:
# #     p = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/t2m/{x}/t2m/'.format(x=x))
# #     p1 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/u10/{x}/u10/'.format(x=x))
# #     p2 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/v10/{x}/v10/'.format(x=x))
# #     p3 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/z/{x}/z/'.format(x=x))
# #     p4 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/t/{x}/t/'.format(x=x))
# #     p5 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tcc/{x}/tcc/'.format(x=x))
# #     p6 = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tp/{x}/tp/'.format(x=x))
#
