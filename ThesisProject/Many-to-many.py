import argparse
import os.path
import torch.nn.init
import torchvision.transforms
from UNET import *
from PrepareDataset import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import zarr
import xarray as xr
import mlflow
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Many-to-many prediction Model")
parser.add_argument(
    "--batch-size",
    type=int,
    default=10,
    metavar="N",
    help="input batch size for training (default:10)")
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=10,
    metavar="N",
    help="input batch size for training (default:10)")
parser.add_argument(
    "--epochs", type=int, default=60, metavar="N", help="number of epochs to train (default:50)"
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
parser.add_argument(
    "--test",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True just to see test results on already trained model"
)
parser.add_argument(
    "--load-model",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True to use default trained model"
)
parser.add_argument(
    "--save-trained-model",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True to save model"
)
parser.add_argument(
    "--draw-graph",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True to draw loss curves for train and validation data"
)


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_epochs = 0


train_dataset = PrepareDataset.TrainDataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataset =  PrepareDataset.ValDataset()
val_loader = torch.utils.data.DataLoader(val_dataset)
test_dataset = PrepareDataset.TestDataset()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)

model = UNet(10, 7).to(device)
model.apply(init_weights)

# Loss type and hyperparameters:
loss_type = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

# use saved model states:
if(args.test or args.load_model):
    state = torch.load('/home/ge75tis/Desktop/oezyurt/model/10_7_UNET/no_dropout_model_lr5_60_epochs')
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    total_epochs = state['epoch']

y_loss = {}
y_loss['train'] = []
y_loss['val'] = []
x_epoch = []


if(args.draw_graph):
    fig = plt.figure(figsize=(36, 12))
    ax0 = fig.add_subplot(121, title="loss")
def draw_loss_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'ro-', label='train', markevery=5)
    ax0.plot(x_epoch, y_loss['val'], 'bo-', label='val', markevery=5)
    if current_epoch == 0:
        ax0.legend()
    fig.savefig('/home/ge75tis/Desktop/loss_new.jpg', dpi=100)


def train(epoch):
    model.train()
    train_loss = 0
    for i, (inputs) in enumerate(train_loader):
        input = inputs.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(input)
        loss = loss_type(outputs, input[:, :7])
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(inputs)

    if(epoch >= 35 and epoch % 5 == 0):
        scheduler.step()

    avg_train_loss = train_loss/len(train_dataset)
    print(f'Epoch: {total_epochs}, Average Training Loss: {avg_train_loss:.6f}')
    y_loss['train'].append(avg_train_loss)

    if (args.draw_graph):
        draw_loss_curve(total_epochs)
    log_scalar("train_loss", avg_train_loss)




test_param_loss = [0 for i in range(7)]
test_avg_param_loss = [0 for i in range(7)]
val_param_loss = [0 for i in range(7)]
val_avg_param_loss = [0 for i in range(7)]
labels = ["T2M", "U10", "V10", 'Z', 'T', "TCC", "TP"]


def val():
    val_loss = 0
    clm_loss = 0
    model.eval()

    for i, (vals) in enumerate(val_loader):
        vals = vals.to(device, dtype=torch.float32)

        # Reduced total loss
        pred = model(vals)
        loss_v = loss_type(pred, vals[:, :7])
        val_loss += loss_v.item() * len(vals)

        # per parameter loss
        for j in range(7):
            val_param_loss[j] += loss_type(pred.select(dim=1, index=j), vals.select(dim=1, index=j)) * len(vals)

    avg_val_loss = val_loss/len(val_dataset)
    for i in range(7):
        val_avg_param_loss[i] = val_param_loss[i] / len(val_dataset)

    print(f'Average Validation Loss: {avg_val_loss:.6f}')
    for i in range(7):
        print(f'Val Loss for {labels[i]}: {val_avg_param_loss[i]:.6f}')

    y_loss['val'].append(avg_val_loss)

    if(args.draw_graph):
        draw_loss_curve(total_epochs)
    log_scalar("val_loss", avg_val_loss)



def test():
    test_loss = 0
    model.eval()
    for i, (test) in enumerate(test_loader):
        test = test.to(device, dtype=torch.float32)
        pred = model(test)
        loss_t = loss_type(pred, test[:, :7])
        test_loss += loss_t.item() * len(test)

        for j in range(7):
           test_param_loss[j] += loss_type(pred.select(dim=1, index=j), test.select(dim=1, index=j)) * len(test)

    avg_test_loss = test_loss/len(test_dataset)
    for i in range(7):
        test_avg_param_loss[i] = test_param_loss[i] / len(test_dataset)

    print(f'Average Test Loss at the End: {avg_test_loss:.6f}')
    for i in range(7):
        print(f'Test Loss for {labels[i]}: {test_avg_param_loss[i]:.6f}')

    log_scalar("test_loss", avg_test_loss)



def log_scalar(name, value):
    mlflow.log_metric(name, value)


"Training, Testing and Validation"
with mlflow.start_run() as run:
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    if(args.test == False):
        for epoch in range(args.epochs):
            train(epoch)
            val()
            total_epochs += 1

    test()
    test_per_parameter()


if(args.save_trained_model):
    state = {
        'epoch': total_epochs + args.epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(state, '/home/ge75tis/Desktop/oezyurt/model/10_7_UNET/nodrop_scheduler_60_epochs')
