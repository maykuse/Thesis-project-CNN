import torch.nn.init
import torchvision.transforms
from UNET import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import zarr
import xarray as xr
import mlflow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 60
batch_size = 10
learning_rate = 0.0005
total_epochs = 0

train_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_train/')
validation_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_val/')
test_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_test/')
# order of the parameters are: t2m, u10, v10, z, t, tcc, tp, lsm, orog, slt


norm = transforms.Normalize((2.78415200e+02, -1.00402647e-01,  2.20140679e-01,  5.40906312e+04,
                                2.74440506e+02,  6.76697789e-01,  9.80986749e-05,  3.37078289e-01,
                                3.79497583e+02,  6.79204298e-01),
                            (2.11294838e+01, 5.57168569e+00, 4.77363485e+00, 3.35202722e+03,
                            1.55503555e+01, 3.62274453e-01, 3.57928990e-04, 4.59003773e-01,
                            8.59872249e+02, 1.16888408e+00))


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


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)

train_dataset = TrainDataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset =  ValDataset()
val_loader = torch.utils.data.DataLoader(val_dataset)

test_dataset = TestDataset()
test_loader = torch.utils.data.DataLoader(test_dataset)

model = UNet(7,1).to(device)
model.apply(init_weights)

loss_type = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# state = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/one_output_lr0005_epoch10')
# model.load_state_dict(state['state_dict'])
# optimizer.load_state_dict(state['optimizer'])
# total_epochs = state['epoch']

# The model architecture is always 6 inputs predicting the 1 remaining output
# changing the indices will change the input/output combinations of the model


def train(epoch):
    model.train()
    train_loss = 0
    for i, (inputs) in enumerate(train_loader):
        input = inputs.index_select(dim=1, index=indices_u10).to(device, dtype=torch.float32)
        target = inputs.index_select(dim=1, index=u10_out).to(device, dtype=torch.float32)

        output = model(input)
        loss = loss_type(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(inputs)

    avg_train_loss = train_loss/len(train_dataset)
    print(f'Epoch: {total_epochs}, Average Training Loss: {avg_train_loss:.6f}')
    log_scalar("u10_individual_train_loss", avg_train_loss)


def val():
    val_loss = 0
    model.eval()
    for i, (vals) in enumerate(val_loader):
        val = vals.index_select(dim=1, index=indices_u10).to(device, dtype=torch.float32)
        target = vals.index_select(dim=1, index=u10_out).to(device, dtype=torch.float32)

        output = model(val)
        loss1 = loss_type(output, target)

        val_loss += loss1.item()

    avg_val_loss = val_loss/len(val_dataset)
    print(f'Average Validation Loss: {avg_val_loss:.6f}')
    log_scalar("u10_individual_val_loss", avg_val_loss)


def test():
    test_loss = 0
    model.eval()
    for i, (tests) in enumerate(test_loader):
        test = tests.index_select(dim=1, index=indices_u10).to(device, dtype=torch.float32)
        target = target.index_select(dim=1, index=u10_out).to(device, dtype=torch.float32)

        output = model(test)
        loss_t = loss_type(output, target)
        test_loss += loss_t.item()

    avg_test_loss = test_loss/len(test_dataset)
    print(f'Average Test Loss at the End: {avg_test_loss:.6f}')

    log_scalar("v10 _individual_test_loss", avg_test_loss)


def log_scalar(name, value):
    mlflow.log_metric(name, value)


for epoch in range(num_epochs):
    train(epoch)
    val()
    total_epochs += 1

test()