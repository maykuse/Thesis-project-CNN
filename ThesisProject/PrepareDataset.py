from UNET import *
from torch.utils.data import Dataset
from torchvision import transforms
import zarr
import xarray as xr


"Open the zarr format resampled dataset"
train_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_train/')
validation_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_val/')
test_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_test/')
# Order of the parameters are as follows: t2m, u10, v10, z, t, tcc, tp, lsm, orog, slt


# Mean and Std of the training set has been computed once and used for the normalization of each set
norm = transforms.Normalize((2.78415200e+02, -1.00402647e-01,  2.20140679e-01,  5.40906312e+04,
                                2.74440506e+02,  6.76697789e-01,  9.80986749e-05,  3.37078289e-01,
                                3.79497583e+02,  6.79204298e-01),
                            (2.11294838e+01, 5.57168569e+00, 4.77363485e+00, 3.35202722e+03,
                            1.55503555e+01, 3.62274453e-01, 3.57928990e-04, 4.59003773e-01,
                            8.59872249e+02, 1.16888408e+00))


# Datasets are transformed into torch tensors and normalized
xr_train = xr.DataArray(train_set)
np_train = xr_train.to_numpy()
torch_train = torch.from_numpy(np_train)
norm_train = norm(torch_train)

xr_val = xr.DataArray(validation_set)
np_val = xr_val.to_numpy()
torch_val = torch.from_numpy(np_val)
norm_val = norm(torch_val)

xr_test = xr.DataArray(test_set)
np_test = xr_test.to_numpy()
torch_test = torch.from_numpy(np_test)
norm_test = norm(torch_test)


class TrainDataset(Dataset):
    def __init__(self):
        self.norm_train = norm_train

    def __getitem__(self, item):
        return self.norm_train[item]

    def __len__(self):
        return len(norm_train)


class ValDataset(Dataset):
    def __init__(self):
        self.norm_val = norm_val

    def __getitem__(self, item):
        return self.norm_val[item]

    def __len__(self):
        return len(norm_val)


class TestDataset(Dataset):
    def __init__(self):
        self.norm_test = norm_test

    def __getitem__(self, item):
        return self.norm_test[item]

    def __len__(self):
        return len(norm_test)