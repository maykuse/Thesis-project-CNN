import pandas as pd
import torch.nn.init
import torchvision.transforms
from UNET import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import zarr
import xarray as xr


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 50
batch_size = 10
learning_rate = 0.001
total_epochs = 0

dataset = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_train/')
validation_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_val/')
# order of the parameters are: t2m, u10, v10, z, t, tcc, tp, lsm, orog, slt

# the mean of the tcc for training and validation data is very different. approx. 10*3 difference
norm = transforms.Normalize((2.78415200e+02, -1.00402647e-01,  2.20140679e-01,  5.40906312e+04,
                                2.74440506e+02,  6.76697789e-01,  9.80986749e-05,  3.37078289e-01,
                                3.79497583e+02,  6.79204298e-01),
                            (2.11294838e+01, 5.57168569e+00, 4.77363485e+00, 3.35202722e+03,
                            1.55503555e+01, 3.62274453e-01, 3.57928990e-04, 4.59003773e-01,
                            8.59872249e+02, 1.16888408e+00))

val_norm = transforms.Normalize((2.78418572e+02, -1.55109431e-01,  2.18645306e-01,  5.41776321e+04,
                                2.74680580e+02,  1.37710218e+02,  9.75816323e-05,  3.37078289e-01,
                                3.79497583e+02,  6.79204298e-01), (2.10886891e+01, 5.54832859e+00, 4.79282050e+00, 3.33387113e+03,
                                1.56281437e+01, 1.37485814e+02, 3.90645171e-04, 4.59003773e-01,
                                8.59872249e+02, 1.16888408e+00))

xr_val = xr.DataArray(validation_set)
np_val = xr_val.to_numpy()
torch_val = torch.from_numpy(np_val)
norm_val = val_norm(torch_val)
selection = val_norm(torch_val)

xrdataset = xr.DataArray(dataset)
np_array = xrdataset.to_numpy()
torch_dataset = torch.from_numpy(np_array)
norm_dataset = norm(torch_dataset)
# You should also think about saving the normalized data as zarr file for more efficient data loading

# selection[:, 0, :, :] = 0   # T2M
# selection[:, 1, :, :] = 0   # U10
# selection[:, 2, :, :] = 0   # V10
# selection[:, 3, :, :] = 0     # Z
# selection[:, 4, :, :] = 0   # T
# selection[:, 5, :, :] = 0   # TCC
# selection[:, 6, :, :] = 0   # TP


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

class SelectionSet(Dataset):
    def __init__(self):
        self.selection = selection
    def __getitem__(self, item):
        return self.selection[item]
    def __len__(self):
        return len(selection)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)


train_dataset = TrainDataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset =  ValDataset()
val_loader = torch.utils.data.DataLoader(val_dataset)

sel_dataset = SelectionSet()
sel_loader = torch.utils.data.DataLoader(sel_dataset)

model = UNet(10, 7).to(device)
model.apply(init_weights)

loss_type = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# use saved model states to resume training:
# state = torch.load('/home/ge75tis/Desktop/oezyurt/model/8192_model')
# model.load_state_dict(state['state_dict'])
# optimizer.load_state_dict(state['optimizer'])
# total_epochs = state['epoch']


for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for i, (inputs) in enumerate(train_loader):
        inputs = inputs.to(device, dtype=torch.float)

        outputs = model(inputs)
        loss = loss_type(outputs, inputs[:, :7])

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
        loss1 = loss_type(target, vals[:, :7])
        val_loss += loss1.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Validation loss: {val_loss/len(val_loader):.6f}')


print('Training finished!')


# selection_loss = 0
# model.eval()
# for (sels, vals) in zip(sel_loader, val_loader):
#     sels = sels.to(device, dtype=torch.float)
#     vals = vals.to(device, dtype=torch.float)
#     target = model(sels)
#     loss2 = loss_type(target, vals)
#     selection_loss += loss2.item()
#
# print(f'Average Selection Loss:  {selection_loss/len(sel_loader):.6f}')


# state = {
#     'epoch': total_epochs + num_epochs,
#     'state_dict': model.state_dict(),
#     'optimizer': optimizer.state_dict()
# }
#
# torch.save(state, '/home/ge75tis/Desktop/oezyurt/model/8192_model')
# print(total_epochs)