import zarr
import xarray as xr
import numpy as np
from torchvision import transforms
from UNET import *
from torch.utils.data import Dataset, DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10
batch_size = 10
learning_rate = 0.0005
total_epochs = 0

validation_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_val/')

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

# selection[:, 0, :, :] = 0   # T2M
selection[:, 1, :, :] = 0   # U10
selection[:, 2, :, :] = 0   # V10
selection[:, 3, :, :] = 0   # Z
selection[:, 4, :, :] = 0   # T
selection[:, 5, :, :] = 0   # TCC
selection[:, 6, :, :] = 0   # TP

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

val_dataset =  ValDataset()
val_loader = torch.utils.data.DataLoader(val_dataset)

sel_dataset = SelectionSet()
sel_loader = torch.utils.data.DataLoader(sel_dataset)

model = UNet(10, 7).to(device)

loss_type = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# state = torch.load('/home/ge75tis/Desktop/oezyurt/model/no_dropout_model_lr5_40_epochs')
# model.load_state_dict(state['state_dict'])
# optimizer.load_state_dict(state['optimizer'])
# total_epochs = state['epoch']

selection_loss = 0
model.eval()
for (sels, vals) in zip(sel_loader, val_loader):
    sels = sels.to(device, dtype=torch.float)
    vals = vals.to(device, dtype=torch.float)
    target = model(sels)
    loss2 = loss_type(target, vals[:, :7])
    selection_loss += loss2.item()

print(f'Average Selection Loss:  {selection_loss/len(sel_loader):.6f}')