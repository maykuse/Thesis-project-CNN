import numpy
import array
import torch.nn.init
import torchvision.transforms
import argparse
from argparse import RawTextHelpFormatter
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

parser = argparse.ArgumentParser(description="Many-to-one prediction Model", formatter_class=RawTextHelpFormatter)
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
    help="input batch size for testing (default:10)")
parser.add_argument(
    "--train-own",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True to train your own model. When (Default:False), already trained models are used for evaluation.\n"
         "When training you can only observe reconstruction losses!"
)
parser.add_argument(
    "--month",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True to categorize losses by month"
)
parser.add_argument(
    "--save-model",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True save your trained model"
)
parser.add_argument(
    "--val",
    type=bool,
    default=True,
    metavar="Bool",
    help="Set False to see results of Test Dataset instead of Validation dataset"
)
parser.add_argument(
    "--output-par",
    type=int,
    default=0,
    choices=range(0,7),
    metavar="[0-6]",
    help="If you set train=True please choose the output parameter.\n"
         "0-T2M    1-U10    2-V10    3-Z    4-T    5-TCC    6-TP"
)
parser.add_argument(
    "--epochs", type=int, default=24, metavar="N", help="number of epochs to train (default:20)"
)
parser.add_argument(
    "--lr", type=float, default=0.0005, metavar="LR", help="learning rate (default: 0.001)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--result-type",
    type=int,
    default=1,
    choices=range(0,6),
    metavar="[0-5]",
    help="Choose what type of results you want to observe (Default:1)\n"
         "0) Reconstruction Losses of models trained WITHOUT DROPOUT\n" 
         "1) Reconstruction Losses of models trained WITH DROPOUT\n"
         "2) Global Gradients in p_all setting\n"
         "3) Global Gradients in p_x, p_other setting\n"
         "4) Spatial Gradients in p_all setting\n"
         "5) Relevances via LRP\n"
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

modelt = UNet(6, 1).to(device)
modelt.apply(init_weights)
loss_type = nn.L1Loss()
loss_type_nored = nn.L1Loss(reduction='none')
optimizer = torch.optim.Adam(modelt.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.25)

models = numpy.empty(7, dtype=UNet)
state = numpy.empty(7, dtype=dict)

if (args.result_type == 0):
    for i in range(7):
        models[i] = UNet(6, 1).to(device)

    for i, j in zip(range(7), params):
        state[i] = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET_TRUE/{parameter}'.format(parameter=j))

    for i in range(7):
        models[i].load_state_dict(state[i]['state_dict'])

y_loss = {}
y_loss['train'] = []
y_loss['val'] = []
y_loss['test'] = []
x_epoch = []


def train(epoch, out_par=0):
    modelt.train()
    train_loss = 0
    for i, (inputs) in enumerate(train_loader):
        input = inputs.index_select(dim=1, index=input_indices[out_par]).to(device, dtype=torch.float32)
        target = inputs.index_select(dim=1, index=out_index[out_par]).to(device, dtype=torch.float32)

        # Random amounts of(chosen from uniform d.) Gaussian Noise is added to the training inputs
        p = np.empty([len(input), 6])
        for j in range(len(input)):
            p[j] = np.random.uniform(low=np.nextafter(0, 1), high=np.nextafter(1, 0), size=6)
        tens_p = torch.tensor(p).to(device, dtype=torch.float32)
        if(args.result_type != 0):
           input = UNET_parts.gaussian_dropout(input, tens_p)

        optimizer.zero_grad()
        output = modelt(input)
        loss = loss_type(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(inputs)


    if (epoch >= 8 and epoch % 4 == 0):
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

avg_val_loss_gridded_m = [[[0 for l in range(12)] for k in range(6)] for j in range(7)]
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


rec_losses = torch.zeros([2, 7, 6])
std_tensor = torch.zeros([6, 730])


multi_parameter = False
see_lrp = False


np.set_printoptions(suppress=True)
def val_or_test(out_par=0):
    counter = 0
    val_loss1 = 0
    index_month = 0
    global results

    if(args.val):
        txt = "val"
        loader = val_loader
    else:
        txt = "test"
        loader = test_loader

    # If models are being trained, val/test loss (without noisy input) is given for comparison
    if(args.train_own):
        valid_loss = 0
        for i, (inputs) in enumerate(loader):
            input = inputs.index_select(dim=1, index=input_indices[out_par]).to(device, dtype=torch.float32)
            target = inputs.index_select(dim=1, index=out_index[out_par]).to(device, dtype=torch.float32)

            output = modelt(input)
            loss = loss_type(output, target)
            valid_loss += loss.item() * len(inputs)

        avg_val_loss = valid_loss / len(loader)
        print(f'Epoch: {total_epochs},', 'Average {t} Loss:'.format(t=txt), f'{avg_val_loss:.6f}')
        log_scalar("{par}_dropout_{t}_loss".format(par=params[out_par], t=txt), avg_val_loss)
        y_loss[txt].append(avg_val_loss)
        results = avg_val_loss

    # See results for already trained models
    else:
        # NO Input Dropout
        if (args.result_type == 0):
            for i in range(7):
                models[i].eval()

            for i, (vals) in enumerate(loader):
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
                val_loss[j] = val_loss[j] / len(loader)
            results = val_loss
            print('Average {t} Losses:'.format(t=txt), f'{results}')


        else:
            # Load each dropout model for evaluation
            for k in range(7):
                lrp_map = torch.zeros([1, 6, 32, 64]).to(device, dtype=torch.float32)
                model = UNet(6, 1).to(device)
                train_state = torch.load(
                        '/home/ge75tis/Desktop/oezyurt/model/6_1_UNET_TRUE_DROPOUT/{par}'.format(par=params[k]))
                model.load_state_dict(train_state['state_dict'])
                optimizer.load_state_dict(train_state['optimizer'])
                total_epochas = train_state['epoch']
                print(f'{params[k]}')

                # In global correlation analysis, there is a single dropout rate for each parameter
                # otherwise a dropout rate for each pixel of each parameter is defined (spatial analysis)
                if(args.result_type == 2 or args.result_type == 3):
                    p_gradients = torch.zeros(6).to(device, dtype=torch.float32)
                    p = torch.zeros(6, requires_grad=True).to(device, dtype=torch.float32)
                    p.fill_(0.999)
                if(args.result_type == 4):
                    p_gradients = torch.zeros([6,32,64]).to(device, dtype=torch.float32)
                    p = torch.zeros([6, 32, 64], requires_grad=True).to(
                        device, dtype=torch.float32)
                    p.fill_(0.9)

                # global p_other and p_x setting:
                # dropout rate of one input parameter is modified to observe global multi-parameter relations
                if(args.result_type == 3):
                    for l in range(6):
                        p[l] = 0.00001

                        for i, (vals) in enumerate(loader):
                            valid = vals.index_select(dim=1, index=input_indices[k]).to(device, dtype=torch.float32)
                            target = vals.index_select(dim=1, index=out_index[k]).to(device, dtype=torch.float32)

                            input = UNET_parts.gaussian_dropout(valid, p)
                            output = model(input)
                            loss = loss_type(output, target)
                            val_loss1 += loss.item()
                            std_tensor[i] = loss.item()

                            # categorize the individual losses by month
                            if (args.month):
                                if (counter in inc_point):
                                    index_month += 1
                                if (counter == 365):
                                    counter = 0
                                    index_month = 0
                                avg_val_loss_gridded_m[k][l][index_month] += loss.item()

                            counter += 1

                        if(args.month):
                            for i in range(12):
                                if (num_of_days_monthly[i] == 31):
                                    avg_val_loss_gridded_m[k][l][i] = avg_val_loss_gridded_m[k][l][i] / 62
                                elif (num_of_days_monthly[i] == 30):
                                    avg_val_loss_gridded_m[k][l][i] = avg_val_loss_gridded_m[k][l][i] / 60
                                elif (i == 1):
                                    avg_val_loss_gridded_m[k][l][i] = avg_val_loss_gridded_m[k][l][i] / 56

                        # save the loss and std for each setting of each parameter
                        rec_losses[0][k][l] = val_loss1 / len(loader)
                        val_loss1 = 0
                        rec_losses[1][k][l] = torch.std(std_tensor)



                # the setting p_all
                else:
                    for i, (vals) in enumerate(loader):
                        valid = vals.index_select(dim=1, index=input_indices[k]).to(device, dtype=torch.float32)
                        target = vals.index_select(dim=1, index=out_index[k]).to(device, dtype=torch.float32)

                        input = valid
                        if(args.result_type == 2):
                            input = UNET_parts.gaussian_dropout(valid, p)
                        if (args.result_type == 4):
                            input = UNET_parts.gaussian_dropout_image(valid, p)


                        output = model(input)
                        if(args.result_type == 4):
                            loss = loss_type_nored(output, target)
                        else:
                            loss = loss_type(output, target)
                        val_loss[k] += loss.item()

                        # compute the gradients of the loss with respect to the dropout rates (p_1D or p_3D)
                        if (args.result_type == 2):
                            loss.backward(inputs=p, retain_graph=True)
                            for j in range(6):
                                std_tensor[j][i] = p_gradients[j]
                        else:
                            loss = torch.squeeze(loss)
                            loss[16][32].backward(inputs=p, retain_graph=True)

                        # Add gradients of input data. You can either take their average later or look at the sum
                        p_gradients.add_(p.grad)
                        p.grad = None

                        # See the sensitivity map produced via LRP for target pixel                        
                        if(args.result_type == 5):
                            lrp = captum.attr.LRP(model)
                            attribution = lrp.attribute(valid, target=(0, 16, 32))
                            lrp_map.add_(attribution)

                        optimizer.zero_grad()

                    val_loss[k] = val_loss[k] / len(loader)

                    # Scale or modify the results to improve visualization
                    # p_gradients = torch.abs(p_gradients)

        if(args.result_type == 1):
            if (args.monthly):
                results = avg_val_loss_gridded_m
            else:
                results = val_loss
        elif(args.result_type == 5):
            results = np.array(lrp_map.detach().cpu().clone().numpy())
        elif(args.result_type == 2 or args.result_type == 4):
            results = np.array(p_gradients.cpu())

    return results



def log_scalar(name, value):
    mlflow.log_metric(name, value)


with mlflow.start_run() as run:
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    mlflow.log_param('batch_size', args.batch_size)
    mlflow.log_param('lr', args.lr)
    mlflow.log_param('total_epochs', args.epochs)

    if(args.train_own):
        for epoch in range(args.epochs):
            train(epoch, args.output_par)
            val_or_test(args.output_par)
            total_epochs += 1
        print("Training finished. Final values:")

    val_or_test(args.output_par)


if(args.save_model):
    train_state = {
        'epoch': args.epochs,
        'state_dict': modelt.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(train_state, '/home/ge75tis/Desktop/oezyurt/model/6_1_UNET_TRUE_DROPOUT/{par}'.format(par=params[args.output_par]))
