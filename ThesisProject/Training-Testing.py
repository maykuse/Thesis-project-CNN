import argparse
import torch.nn.init
import torchvision.transforms
from UNET import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import zarr
import xarray as xr
import mlflow
import pickle

parser = argparse.ArgumentParser(description="Meteorological Parameter Prediction Model")
parser.add_argument(
    "--batch-size",
    type=int,
    default=10,
    metavar="N",
    help="input batch size for training (default:10)")
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=2922,
    metavar="N",
    help="input batch size for training (default:2922)")
parser.add_argument(
    "--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default:10)"
)
parser.add_argument(
    "--lr", type=float, default=0.0005, metavar="LR", help="learning rate (default: 0.0005)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# # You should also think about saving the normalized data as zarr file for more efficient data loading


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


# def init_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#         torch.nn.init.xavier_uniform_(m.weight)
#         # m.bias.data.fill_(0.01)


train_dataset = TrainDataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

val_dataset =  ValDataset()
val_loader = torch.utils.data.DataLoader(val_dataset)

test_dataset = TestDataset()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size)

model = UNet(10, 7).to(device)
# model.apply(init_weights)

# Try L2 Loss, add script for interactive argument selection, all parameters lr, batchsize..
loss_type = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# # use saved model states to resume training:
state = torch.load('/home/ge75tis/Desktop/oezyurt/model/no_dropout_model_lr5_60_epochs')
model.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])
total_epochs = state['epoch']


def train(epoch):
    model.train()
    for i, (inputs) in enumerate(train_loader):
        inputs = inputs.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_type(outputs, inputs[:, :7])
        loss.backward()
        optimizer.step()

    print(f'Epoch: {total_epochs + epoch}, Average Training Loss: {loss.data.item():.6f}')

    log_scalar("train_loss", loss.data.item())


def val():
    val_loss = 0
    model.eval()
    for i, (vals) in enumerate(val_loader):
        vals = vals.to(device, dtype=torch.float32)
        pred = model(vals)
        val_loss = loss_type(pred, vals[:, :7])

    print(f'Average Validation Loss: {val_loss.data.item():.6f}')

    log_scalar("val_loss", val_loss)

def test():
    test_loss = 0
    model.eval()
    for i, (test) in enumerate(test_loader):
        test = test.to(device, dtype=torch.float32)
        pred = model(test)
        test_loss = loss_type(pred, test[:, :7])

    print(f'Average Test Loss at the End: {test_loss.data.item():.6f}')

    log_scalar("test_loss", test_loss)


def test_per_parameter():
    model.eval()
    for i, (test) in enumerate(test_loader):
        test = test.to(device, dtype=torch.float32)
        pred = model(test)
        t2m_loss = loss_type(pred.select(dim=1, index=0), test.select(dim=1, index=0))
        u10_loss = loss_type(pred.select(dim=1, index=1), test.select(dim=1, index=1))
        v10_loss = loss_type(pred.select(dim=1, index=2), test.select(dim=1, index=2))
        z_loss = loss_type(pred.select(dim=1, index=3), test.select(dim=1, index=3))
        t_loss = loss_type(pred.select(dim=1, index=4), test.select(dim=1, index=4))
        tcc_loss = loss_type(pred.select(dim=1, index=5), test.select(dim=1, index=5))
        tp_loss = loss_type(pred.select(dim=1, index=6), test.select(dim=1, index=6))

    print(f'Test Loss for T2M: {t2m_loss.data.item():.6f}')
    print(f'Test Loss for U10: {u10_loss.data.item():.6f}')
    print(f'Test Loss for V10: {v10_loss.data.item():.6f}')
    print(f'Test Loss for Z: {z_loss.data.item():.6f}')
    print(f'Test Loss for T: {t_loss.data.item():.6f}')
    print(f'Test Loss for TCC: {tcc_loss.data.item():.6f}')
    print(f'Test Loss for TP: {tp_loss.data.item():.6f}')


def log_scalar(name, value):
    mlflow.log_metric(name, value)




# Training, Testing, Validation:
with mlflow.start_run() as run:
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    # for epoch in range(args.epochs):
    #     train(epoch)
    #     val()

    test()
    test_per_parameter()

# state = {
#     'epoch': total_epochs + args.epochs,
#     'state_dict': model.state_dict(),
#     'optimizer': optimizer.state_dict()
# }
#
# torch.save(state, '/home/ge75tis/Desktop/oezyurt/model/no_dropout_model_lr5_60_epochs')
# print(total_epochs)


# With Learning rate 0.0005 seems to overfit at  epochs
# Avg. Training Loss: 0.055 at 60 epoch, Minimum Validation Loss: 0.023

# Loss per parameter then comparing with Climatology method of monthly standard deviations.
# 3 periods, yearly, monthly, weekly
# Scale of the data should be the same
# Think about seasons and different periods