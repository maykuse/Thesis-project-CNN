import pandas as pd
import torch.nn.init
import torchvision.transforms

from UNET import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import zarr
import xarray as xr
import wxee
import datetime
import pandas

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
batch_size = 5
learning_rate = 0.05


### new = wxee.xarray.DataArrayAccessor(xrdataset).normalize()
### test_array = new.to_numpy()
### norm_dataset = torch.from_numpy(test_array)

dataset = zarr.create((8784, 7, 32, 64))
dataset[:,0,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/t2m/1980/t2m/')
dataset[:,1,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/u10/1980/u10/')
dataset[:,2,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/v10/1980/v10/')
dataset[:,3,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/z/1980/z/')
dataset[:,4,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/t/1980/t/')
dataset[:,5,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tcc/1980/tcc/')
dataset[:,6,:,:] = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/TRAIN_DATA/tp/1980/tp/')

norm = transforms.Normalize((2.78382787e+02, -8.24015036e-02, 2.22236745e-01, 5.40970992e+04, 2.74478000e+02, 6.73734047e-01, 9.93376168e-05),
                            ())

# dti = pd.date_range("1980-01-01", periods=732 ,freq="12H")
# print(dti)
# xrdataset = xr.DataArray(dataset)
# panda = xrdataset.to_pandas()
# panda.resample('12H')


xrdataset = xr.DataArray(dataset)
np_array = xrdataset.to_numpy()
dataset = torch.from_numpy(np_array)
norm_dataset = norm(dataset)


class TestDataset(Dataset):
    def __init__(self):
        self.dataset = norm_dataset

    def __getitem__(self, param):
        return self.dataset[param]

    def __len__(self):
        return len(dataset)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

train_dataset = TestDataset()
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

model = UNet(7,7).to(device)
model.apply(init_weights)

loss_type = nn.L1Loss()
# optimizer algorithm may be changed to see different results
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)


for epoch in range(num_epochs):
    for i, (inputs) in enumerate(train_loader):
        inputs = inputs.to(device, dtype=torch.float)

        outputs = model(inputs)
        loss = loss_type(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Training finished!')

