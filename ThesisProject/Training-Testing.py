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

num_epochs = 1
batch_size = 5
learning_rate = 0.2
total_epochs = 0

dataset = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/RESAMPLED_TRAIN_DATA/resample.zarr/')

norm = transforms.Normalize((2.78396666e+02, -9.47355324e-02, 2.26018299e-01, 5.40951497e+04, 2.74467023e+02, 6.74842584e-01, 1.00444445e-04),
                            (2.12417626e+01, 5.55599043e+00, 4.77692863e+00, 3.35398112e+03, 1.55931385e+01, 3.62925103e-01, 3.68333206e-04))

xrdataset = xr.DataArray(dataset)
np_array = xrdataset.to_numpy()
torch_dataset = torch.from_numpy(np_array)
norm_dataset = norm(torch_dataset)


class TestDataset(Dataset):
    def __init__(self):
        self.norm_dataset = norm_dataset

    def __getitem__(self, param):
        return self.norm_dataset[param]

    def __len__(self):
        return len(norm_dataset)

# def init_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#         torch.nn.init.xavier_uniform_(m.weight)
#         m.bias.data.fill_(0.01)

train_dataset = TestDataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)
model = UNet(7,7).to(device)
# model.apply(init_weights)

# optimizer algorithm may be changed to see different results
loss_type = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# used saved model states to resume training:
state = torch.load('/home/ge75tis/Desktop/oezyurt/model/saved_model')
model.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])
total_epochs = state['epoch']

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

state = {
    'epoch': total_epochs + num_epochs,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
}

torch.save(state, '/home/ge75tis/Desktop/oezyurt/model/saved_model')
print(total_epochs)