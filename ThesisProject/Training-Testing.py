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
batch_size = 10
learning_rate = 0.0001
total_epochs = 0

dataset = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_train/resample.zarr/')
validation_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_val/')

norm = transforms.Normalize((2.78396666e+02, -9.47355324e-02, 2.26018299e-01,
                             5.40951497e+04, 2.74467023e+02, 6.74842584e-01, 1.00444445e-04),
                            (2.12417626e+01, 5.55599043e+00, 4.77692863e+00, 3.35398112e+03,
                             1.55931385e+01, 3.62925103e-01, 3.68333206e-04))

val_norm = transforms.Normalize((2.78706985e+02, -1.46182892e-01,  2.13817276e-01,  5.41781034e+04,
                                 2.74708052e+02,  1.37723293e+02,  1.00084267e-04), (2.12578202e+01, 5.55930170e+00,
                                4.79096587e+00, 3.33503369e+03, 1.56392333e+01, 1.37498117e+02, 3.94158774e-04))

xr_val = xr.DataArray(validation_set)
np_val = xr_val.to_numpy()
torch_val = torch.from_numpy(np_val)
norm_val = val_norm(torch_val)

xrdataset = xr.DataArray(dataset)
np_array = xrdataset.to_numpy()
torch_dataset = torch.from_numpy(np_array)
norm_dataset = norm(torch_dataset)
# You should also think about saving the normalized data as zarr file for more efficient data loading


class TrainDataset(Dataset):
    def __init__(self):
        self.norm_dataset = norm_dataset

    def __getitem__(self, param):
        return self.norm_dataset[param]

    def __len__(self):
        return len(norm_dataset)

class ValDataset(Dataset):
    def __init__(self):
        self.norm_val = norm_val

    def __getitem__(self, item):
        return self.norm_val[item]

    def __len__(self):
        return len(norm_val)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


train_dataset = TrainDataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset =  ValDataset()
val_loader = torch.utils.data.DataLoader(val_dataset)

model = UNet(7, 7).to(device)
model.apply(init_weights)

# optimizer algorithm may be changed to see different results
loss_type = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# used saved model states to resume training:
# state = torch.load('/home/ge75tis/Desktop/oezyurt/model/saved_7bridged_model')
# model.load_state_dict(state['state_dict'])
# optimizer.load_state_dict(state['optimizer'])
# total_epochs = state['epoch']

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for i, (inputs) in enumerate(train_loader):
        inputs = inputs.to(device, dtype=torch.float)

        outputs = model(inputs)
        loss = loss_type(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {train_loss/len(train_loader):.6f}')
    #    scheduler.step()

    val_loss = 0
    model.eval()
    for i, (vals) in enumerate(val_loader):
        vals = vals.to(device, dtype=torch.float)
        target = model(vals)
        loss1 = loss_type(target, vals)
        val_loss += loss1.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Validation loss: {val_loss/len(val_loader):.6f}')



print('Training finished!')

# state = {
#     'epoch': total_epochs + num_epochs,
#     'state_dict': model.state_dict(),
#     'optimizer': optimizer.state_dict()
# }
#
# torch.save(state, '/home/ge75tis/Desktop/oezyurt/model/saved_7bridged_model')
# print(total_epochs)