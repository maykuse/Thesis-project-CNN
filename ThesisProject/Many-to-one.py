import numpy
import array
import torch.nn.init
import torchvision.transforms
import argparse
import UNET_parts
from UNET import *
import PrepareDataset
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import zarr
import xarray as xr
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import pandas as pd
import networkx as nx
import string
import pygraphviz
import scipy
from textwrap import wrap
import math
import captum
from captum.attr._utils.lrp_rules import EpsilonRule

parser = argparse.ArgumentParser(description="Many-to-one prediction Model")
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
    "--output-parameter",
    type=int,
    default=0,
    choices=range(0,7),
    metavar="[0-6]",
    help="Choose the output parameter and model to be trained"
)
parser.add_argument(
    "--epochs", type=int, default=20, metavar="N", help="number of epochs to train (default:20)"
)
parser.add_argument(
    "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
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
    "--draw-graph",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True to draw loss curves for train and validation data"
)
parser.add_argument(
    "--no-dropout",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True to see results of models trained without dropout"
)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_epochs = 0


params = ["t2m", "u10", "v10", 'z', 't', "tcc", "tp"]
params_C = ["T2M", "U10", "V10", 'Z', 'T', "TCC", "TP"]

# all input-output index configurations
input_indices = torch.zeros([7,6], dtype=torch.int)
out_index = torch.zeros([7,1], dtype=torch.int)
for i in range(7):
    out_index[i] = i
    count = 0
    for j in range(7):
        if(i == j):
            continue
        input_indices[i][count] = j
        count += 1


train_dataset = PrepareDataset.TrainDataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

val_dataset = PrepareDataset.ValDataset()
val_loader = torch.utils.data.DataLoader(val_dataset)

test_dataset = PrepareDataset.TestDataset()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)

model = UNet(6, 1).to(device)
model.apply(init_weights)
loss_type = nn.L1Loss()
loss_type_nored = nn.L1Loss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

models = numpy.empty(7, dtype=UNet)
state = numpy.empty(7, dtype=dict)

if (args.no_dropout):
    for i in range(7):
        models[i] = UNet(6, 1).to(device)

    for i, j in zip(range(7), params):
        state[i] = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET_TRUE/{parameter}'.format(parameter=j))

    for i in range(7):
        models[i].load_state_dict(state[i]['state_dict'])

y_loss = {}
y_loss['train'] = []
y_loss['val'] = []
x_epoch = []


def train(epoch, out_par):
    model.train()
    train_loss = 0
    for i, (inputs) in enumerate(train_loader):
        input = inputs.index_select(dim=1, index=input_indices[out_par]).to(device, dtype=torch.float32)
        target = inputs.index_select(dim=1, index=out_index[out_par]).to(device, dtype=torch.float32)

        # Random amounts of(chosen from uniform d.) Gaussian Noise is added to the training inputs
        p = np.empty([len(input), 6])
        for j in range(len(input)):
            p[j] = np.random.uniform(low=np.nextafter(0, 1), high=np.nextafter(1, 0), size=6)
        tens_p = torch.tensor(p).to(device, dtype=torch.float32)
        if(not args.no_dropout):
           input = UNET_parts.gaussian_dropout(input, tens_p)

        optimizer.zero_grad()
        output = model(input)
        loss = loss_type(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(inputs)


    if (epoch >= 10 and epoch % 2 == 0):
        scheduler.step()

    avg_train_loss = train_loss / len(train_dataset)
    print(f'Epoch: {total_epochs}, Average Training Loss: {avg_train_loss:.6f}')
    log_scalar("{par}_dropout_train_loss".format(par=params[out_par]), avg_train_loss)
    y_loss['train'].append(avg_train_loss)



valids = numpy.empty(7, dtype=torch.Tensor)
testy = numpy.empty(7, dtype=torch.Tensor)
targets = numpy.empty(7, dtype=torch.Tensor)
outputs = numpy.empty(7, dtype=torch.Tensor)

losses = numpy.empty(7, dtype=torch.Tensor)
val_loss = [0 for i in range(7)]
test_loss = [0 for i in range(7)]

avg_val_loss_gridded_m = [[[0 for l in range(12)] for k in range(6)] for j in range(7)]
month = False
num_of_days_monthly = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
inc_point = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
index_month = 0


all_labels = [["" for x in range(6)] for y in range(7)]
new_labels = [["" for x in range(5)] for y in range(7)]
best_predictors_6 = [3, 1, 1, 3, 0, 2, 2]
for i in range(7):
    c = 0
    for j in range(6):
        all_labels[i][j] = params[input_indices[i][j]]
        if(j != best_predictors_6[i] and c != 5):
            new_labels[i][c] = params[input_indices[i][j]]
            c += 1


rec_losses = torch.zeros([7, 2, 6])
std_tensor = torch.zeros([6, 730])

# save_figs = False
# gradient_bars = False
# draw_world_map = True
# bar_chart_std = False
oneD_p = True
multi_parameter = False
# draw_grad_bar = False
see_lrp = False


np.set_printoptions(suppress=True)
def val(out_par):
    counter = 0
    val_loss1 = 0
    index_month = 0
    global results


    if (args.no_dropout):
        for i in range(7):
            models[i].eval()

        for i, (vals) in enumerate(val_loader):
            for j in range(7):
                valids[j] = vals.index_select(dim=1, index=input_indices[j]).to(device, dtype=torch.float32)
                targets[j] = vals.index_select(dim=1, index=out_index[j]).to(device, dtype=torch.float32)

            for j in range(7):
                outputs[j] = models[j](valids[j])

            for j in range(7):
                losses[j] = loss_type(outputs[j], targets[j])

            for j in range(7):
                val_loss[j] += losses[j].item()

            counter += 1

        for j in range(7):
            val_loss[j] = val_loss[j] / len(val_dataset)
        print(f'Average Validation Losses: {val_loss}')
        results = val_loss
        
    # grid_x = [0.00001, 0.1, 0.5, 0.999, 0.9999, 0.99995, 0.99999]
    # grid = [0.00001, 0.9, 0.999, 0.9999, 0.99995, 0.99999, 0.999999]
    
    else:
        for k in range(7):
            lrp_map = torch.zeros([1, 6, 32, 64]).to(device, dtype=torch.float32)
            model = UNet(6, 1).to(device)
            train_state = torch.load(
                    '/home/ge75tis/Desktop/oezyurt/model/6_1_UNET_TRUE_DROPOUT/{par}'.format(par=params[k]))
            model.load_state_dict(train_state['state_dict'])
            optimizer.load_state_dict(train_state['optimizer'])
            total_epochs = train_state['epoch']
            print(f'{params[k]}')

            if(oneD_p):
                p_gradients = torch.zeros(6).to(device, dtype=torch.float32)
                p = torch.zeros(6, requires_grad=True).to(device, dtype=torch.float32)
                p.fill_(0.999)
            else:
                p_gradients = torch.zeros([6,32,64]).to(device, dtype=torch.float32)
                p = torch.zeros([6, 32, 64], requires_grad=True).to(
                    device, dtype=torch.float32)
                p.fill_(0.9)

            if(multi_parameter):
                for l in range(6):
                    p[l] = 0.00001

                    for i, (vals) in enumerate(val_loader):
                        valid = vals.index_select(dim=1, index=input_indices[k]).to(device, dtype=torch.float32)
                        target = vals.index_select(dim=1, index=out_index[k]).to(device, dtype=torch.float32)

                        noisy_input = UNET_parts.gaussian_dropout(valid, p)
                        output = model(noisy_input)
                        loss = loss_type(output, target)
                        val_loss1 += loss.item()
                        std_tensor[i] = loss.item()

                        if (month):
                            if (counter in inc_point):
                                index_month += 1
                            if (counter == 365):
                                counter = 0
                                index_month = 0
                            avg_val_loss_gridded_m[k][l][index_month] += loss.item()

                        counter += 1

                    if (bar_chart_std):
                        rec_losses[k][0][l] = val_loss1 / len(val_dataset)
                        val_loss1 = 0
                        rec_losses[k][1][l] = torch.std(std_tensor)
                        results = rec_losses


                    if(month):
                        for i in range(12):
                            if (i == 0 or i == 2 or i == 4 or i == 6 or i == 7 or i == 9 or i == 11):
                                avg_val_loss_gridded_m[k][l][i] = avg_val_loss_gridded_m[k][l][i] / 62
                            elif (i == 3 or i == 5 or i == 8 or i == 10):
                                avg_val_loss_gridded_m[k][l][i] = avg_val_loss_gridded_m[k][l][i] / 60
                            elif (i == 1):
                                avg_val_loss_gridded_m[k][l][i] = avg_val_loss_gridded_m[k][l][i] / 56

                        results = avg_val_loss_gridded_m


            else:
                for i, (vals) in enumerate(val_loader):
                    valid = vals.index_select(dim=1, index=input_indices[k]).to(device, dtype=torch.float32)
                    target = vals.index_select(dim=1, index=out_index[k]).to(device, dtype=torch.float32)

                    if(oneD_p):
                        noisy_input = UNET_parts.gaussian_dropout(valid, p)
                    else:
                        noisy_input = UNET_parts.gaussian_dropout_image(valid, p)


                    output = model(noisy_input)
                    loss = loss_type(output, target)
                    if (oneD_p):
                        loss.backward(inputs=p, retain_graph=True)
                        for j in range(6):
                            std_tensor[j][i] = p_gradients[j]
                    else:
                        loss = torch.squeeze(loss)
                        loss[16][32].backward(inputs=p, retain_graph=True)

                    'Add gradients of input data. You can either take their average later or look at the sum'
                    p_gradients.add_(p.grad)
                    p.grad = None

                    if(see_lrp):
                        lrp = captum.attr.LRP(model)
                        attribution = lrp.attribute(valid, target=(0, 16, 32))
                        lrp_map.add_(attribution)

                    optimizer.zero_grad()

                'Scale or modify the p_gradient to visualize different results'
                # p_gradients = torch.abs(p_gradients)

                if (see_lrp):
                    results = np.array(lrp_map.detach().cpu().clone().numpy())
                else:
                    results = np.array(p_gradients.cpu())

    return results

def test(out_par):
    test_loss1 = 0
    if(args.no_dropout):
        for i in range(7):
            models[i].eval()
            
        for i, (tests) in enumerate(test_loader):
            for j in range(7):
                testy[j] = tests.index_select(dim=1, index=input_indices[j]).to(device, dtype=torch.float32)
                targets[j] = tests.index_select(dim=1, index=out_index[j]).to(device, dtype=torch.float32)

            for j in range(7):
                outputs[j] = models[j](testy[j])

            for j in range(7):
                losses[j] = loss_type(outputs[j], targets[j])

            for j in range(7):
                test_loss[j] += losses[j].item()

        for j in range(7):
            test_loss[j] = test_loss[j] / len(test_dataset)

        print(f'Average Test Losses: {test_loss}')
        # log_scalar("a_individual_test_loss", test_loss)
        
        
    for i, (tests) in enumerate(test_loader):
        test = tests.index_select(dim=1, index=input_indices[out_par]).to(device, dtype=torch.float32)
        target = tests.index_select(dim=1, index=out_index[out_par]).to(device, dtype=torch.float32)

        output = model(test)
        loss = loss_type(output, target)
        test_loss1 += loss.item()
        


def log_scalar(name, value):
    mlflow.log_metric(name, value)


with mlflow.start_run() as run:
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    mlflow.log_param('batch_size', args.batch_size)
    mlflow.log_param('lr', args.lr)
    mlflow.log_param('total_epochs', args.epochs)

    if(not args.test):
        for epoch in range(args.epochs):
            train(epoch, 0)
            # val(0)
            total_epochs += 1

    print("Training finished. Final values:")
    val(0)
    test(0)


save = False
if(save):
    train_state = {
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(train_state, '/home/ge75tis/Desktop/oezyurt/model/5_1_UNET/u10_no_tp')
