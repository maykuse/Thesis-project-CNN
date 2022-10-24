from UNET import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import zarr
import xarray as xr
import wxee

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
batch_size = 10
learning_rate = 0.005

transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

dataset = zarr.create((8760, 7, 32, 64))
dataset[:,0,:,:] = zarr.open("/home/ge75tis/Desktop/oezyurt/zarr dataset/2m_temperature_zarr/1982/t2m/")
dataset[:,1,:,:] = zarr.open("/home/ge75tis/Desktop/oezyurt/zarr dataset/10m_u_component_of_wind_zarr/1982/u10/")
dataset[:,2,:,:] = zarr.open("/home/ge75tis/Desktop/oezyurt/zarr dataset/10m_v_component_of_wind_zarr/1982/v10/")
dataset[:,3,:,:] = zarr.open("/home/ge75tis/Desktop/oezyurt/zarr dataset/geopotential_500_zarr/1982/z/")
dataset[:,4,:,:] = zarr.open("/home/ge75tis/Desktop/oezyurt/zarr dataset/temperature_850_zarr/1982/t/")
dataset[:,5,:,:] = zarr.open("/home/ge75tis/Desktop/oezyurt/zarr dataset/total_cloud_cover_zarr/1982/tcc/")
dataset[:,6,:,:] = zarr.open("/home/ge75tis/Desktop/oezyurt/zarr dataset/total_precipitation_zarr/1982/tp")

xrdataset = xr.DataArray(dataset)
new = wxee.xarray.DataArrayAccessor(xrdataset).normalize()
# How do I set this normalized DataArray back into a tensor or some kind of dataset that works in training?


# norm = transforms.Normalize((2.78151650e+02, -6.46198477e-02, 2.18526315e-01, 5.40422283e+04, 2.74193348e+02, 6.79718728e-01, 9.92141201e-05),
#                             (2.12733154e+01, 5.52872994e+00, 4.74617300e+00, 3.37250806e+03, 1.55878069e+01, 3.59121453e-01, 3.47599315e-04))


class TestDataset(Dataset):
    def __init__(self):
        self.dataset = actual_dataset

    def __getitem__(self, param):
        return self.dataset[param]

    def __len__(self):
        return len(dataset)


train_dataset = TestDataset()
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

model = UNet(7,7).to(device)

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

