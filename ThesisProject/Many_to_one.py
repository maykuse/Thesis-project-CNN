import numpy
import array
import torch.nn.init
import torchvision.transforms
import argparse
from argparse import RawTextHelpFormatter
import Figures
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
    "--train-own",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True to train your own model with input dropout. When (Default:False), already trained models are used for evaluation.\n"
         "When training you can only observe reconstruction losses!"
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
    "--save-model",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True save your trained model"
)
parser.add_argument(
    "--epochs", type=int, default=24, metavar="N", help="number of epochs to train (default:20)"
)
parser.add_argument(
    "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
)
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
    "--log-interval",
    type=int,
    default=100,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--val",
    type=bool,
    default=True,
    metavar="Bool",
    help="Set False to see results of Test Dataset instead of Validation dataset"
)
parser.add_argument(
    "--month",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True to categorize losses by month"
)

parser.add_argument(
    "--figure-opts",
    type=int,
    default=0,
    choices=range(0,6),
    metavar="[0-5]",
    help="If you want to draw a figure, please choose which one:\n"
         "0) Many-to-one vs. Dropout vs. Climatology Reconstruction Losses \n"
         "1) Percentage Loss Decrease Heatmap (p_x ~ 0, p_other ~ 1). Hierarchical/Distance Clusters \n"
         "2) Gradient Bar Chart for p_all setting. Hierarchical/Distance Clusters\n"
         "3) Gradient Bar Chart for p_x, p_other setting\n"
         "4) Gradient World Maps \n"
         "5) LRP World Maps \n"
)

if __name__ == "__main__":
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
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.2)

models_nop = numpy.empty(7, dtype=UNet)
models_yep = numpy.empty(7, dtype=UNet)
state = numpy.empty(7, dtype=dict)
state_yep = numpy.empty(7, dtype=dict)

if (args.figure_opts == 0):
    for i in range(7):
        models_nop[i] = UNet(6, 1).to(device)
        models_yep[i] = UNet(6, 1).to(device)

    for i, j in zip(range(7), params):
        state[i] = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET_TRUE/{parameter}'.format(parameter=j))
        state_yep[i] = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET_TRUE_DROPOUT/{parameter}'.format(parameter=j))

    for i in range(7):
        models_nop[i].load_state_dict(state[i]['state_dict'])
        models_yep[i].load_state_dict(state_yep[i]['state_dict'])

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
        input = UNET_parts.gaussian_dropout(input, tens_p)

        optimizer.zero_grad()
        output = modelt(input)
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
outputs = numpy.empty([2,7], dtype=torch.Tensor)
losses = numpy.empty([2,7], dtype=torch.Tensor)

val_loss = [[0 for i in range(7)] for j in range(2)]
multi_grads = [[0 for i in range(6)] for j in range(7)]

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


rec_losses = torch.zeros([2, 7, 7])
std_tensor = torch.zeros([730])
std_tensor_m = torch.zeros([6, 730])

if(args.figure_opts == 4):
    grads = torch.zeros([7, 6, 32, 64]).to(device, dtype=torch.float32)
else:
    grads = torch.zeros([7, 6]).to(device, dtype=torch.float32)

lrp_map = torch.zeros([7, 6, 32, 64]).to(device, dtype=torch.float32)


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
        if (args.figure_opts == 0):
            for i in range(7):
                models_nop[i].eval()
                models_yep[i].eval()

            for i, (vals) in enumerate(loader):
                for j in range(7):
                    valids[j] = vals.index_select(dim=1, index=input_indices[j]).to(device, dtype=torch.float32)
                    targets[j] = vals.index_select(dim=1, index=out_index[j]).to(device, dtype=torch.float32)

                for j in range(7):
                    outputs[0][j] = models_nop[j](valids[j])
                    outputs[1][j] = models_yep[j](valids[j])

                for j in range(7):
                    losses[0][j] = loss_type(outputs[0][j], targets[j])
                    losses[1][j] = loss_type(outputs[1][j], targets[j])

                for j in range(7):
                    val_loss[0][j] += losses[0][j].item()
                    val_loss[1][j] += losses[1][j].item()

                counter += 1

            for j in range(7):
                val_loss[0][j] = val_loss[0][j] / len(loader)
                val_loss[1][j] = val_loss[1][j] / len(loader)

            results = val_loss
            print('Average {t} Losses\n'.format(t=txt), "trained without Dropout:", f'{results[0]}\n', "trained with Dropout:", f'{results[1]}')


        else:
            # Load each dropout model for evaluation
            for k in range(7):
                model = UNet(6, 1).to(device)
                train_state = torch.load(
                        '/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/{par}_gaussiandropout_35_epochs'.format(par=params[k]))
                model.load_state_dict(train_state['state_dict'])
                optimizer.load_state_dict(train_state['optimizer'])
                total_epochas = train_state['epoch']
                print(f'{params[k]}')

                # In global correlation analysis, there is a single dropout rate for each parameter
                # otherwise a dropout rate for each pixel of each parameter is defined (spatial analysis)
                if(args.figure_opts == 4):
                    p_gradients = torch.zeros([6,32,64]).to(device, dtype=torch.float32)
                    p = torch.zeros([6, 32, 64], requires_grad=True).to(
                        device, dtype=torch.float32)
                    p.fill_(0.9)
                else:
                    p_gradients = torch.zeros(6).to(device, dtype=torch.float32)
                    p = torch.zeros(6, requires_grad=True).to(device, dtype=torch.float32)
                    p.fill_(0.999)

                # global p_other and p_x setting:
                # dropout rate of one input parameter is modified to observe global multi-parameter relations
                if(args.figure_opts == 1 or args.figure_opts == 3):
                    for l in range(7):
                        p_gradients = torch.zeros(6).to(device, dtype=torch.float32)
                        p = torch.zeros(6, requires_grad=True).to(device, dtype=torch.float32)
                        p.fill_(0.999999)
                        if(l < 6):
                            p[l] = 0.000001

                        for i, (vals) in enumerate(loader):
                            valid = vals.index_select(dim=1, index=input_indices[k]).to(device, dtype=torch.float32)
                            target = vals.index_select(dim=1, index=out_index[k]).to(device, dtype=torch.float32)

                            input = UNET_parts.gaussian_dropout(valid, p)
                            output = model(input)
                            loss = loss_type(output, target)
                            val_loss1 += loss.item()
                            std_tensor[i] = loss.item()

                            if(args.figure_opts == 3):
                                loss.backward(inputs=p, retain_graph=True)
                                p_gradients.add_(p.grad)
                                p.grad = None
                                optimizer.zero_grad()

                            # categorize the individual losses by month
                            if (args.month and l < 6):
                                if (counter in inc_point):
                                    index_month += 1
                                if (counter == 365):
                                    counter = 0
                                    index_month = 0
                                avg_val_loss_gridded_m[k][l][index_month] += loss.item()

                            counter += 1

                        if(args.month and l < 6):
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
                        if(l < 6):
                            multi_grads[k][l] = p_gradients

                # the setting p_all
                else:
                    for i, (vals) in enumerate(loader):
                        valid = vals.index_select(dim=1, index=input_indices[k]).to(device, dtype=torch.float32)
                        target = vals.index_select(dim=1, index=out_index[k]).to(device, dtype=torch.float32)

                        if (args.figure_opts == 5):
                            lrp = captum.attr.LRP(model)
                            attribution = lrp.attribute(valid, target=(0, 16, 32))
                            lrp_map[k].add_(attribution)
                            continue

                        if (args.figure_opts == 4):
                            input = UNET_parts.gaussian_dropout_image(valid, p)
                        else:
                            input = UNET_parts.gaussian_dropout(valid, p)

                        output = model(input)
                        if(args.figure_opts == 4):
                            # this depends on the type of world map you want to observe it can also be reduction loss
                            loss = loss_type_nored(output, target)
                        else:
                            loss = loss_type(output, target)

                        # compute the gradients of the loss with respect to the dropout rates (p_1D or p_3D)
                        if (args.figure_opts == 4):
                            loss = torch.squeeze(loss)
                            loss[16][32].backward(inputs=p, retain_graph=True)
                        else:
                            loss.backward(inputs=p, retain_graph=True)
                            for j in range(6):
                                std_tensor_m[j][i] = p_gradients[j]

                        # Add gradients of input data. You can either take their average later or look at the sum
                        p_gradients.add_(p.grad)
                        p.grad = None
                        optimizer.zero_grad()

                    val_loss[1][k] = val_loss[1][k] / len(loader)
                    p_gradients = torch.div(p_gradients, len(loader))
                    grads[k] = p_gradients
                    # Scale or modify the results to improve visualization


        if(args.figure_opts == 0):
            if (args.month):
                results = avg_val_loss_gridded_m
            else:
                results = val_loss
        if(args.figure_opts == 1):
            results = rec_losses.numpy()
        if(args.figure_opts == 2 or args.figure_opts == 4):
            results = np.array(grads.detach().cpu().clone().numpy())
        if(args.figure_opts == 3):
            results = np.array(multi_grads.detach().cpu().clone().numpy())
        if(args.figure_opts == 5):
            results = np.array(lrp_map.detach().cpu().clone().numpy())


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

    figure_values = val_or_test(args.output_par)

    if (args.figure_opts == 0):
        Figures.Loss_Comparison(figure_values)
    if (args.figure_opts == 1):
        Figures.Similarity_Graphs(figure_values, 0)
    if (args.figure_opts == 2):
        Figures.Grad_Bar_Chart(figure_values)
        Figures.Similarity_Graphs(figure_values, 1)
    if (args.figure_opts == 3):
        Figures.Grad_Bar_Chart_Multi(figure_values)
    if (args.figure_opts == 4):
        Figures.Grad_World_Maps(figure_values)
    if (args.figure_opts == 5):
        Figures.LRP_World_Maps(figure_values)

if(args.save_model):
    train_state = {
        'epoch': args.epochs,
        'state_dict': modelt.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(train_state, '/home/ge75tis/Desktop/oezyurt/model/6_1_UNET_TRUE_DROPOUT/{par}'.format(par=params[args.output_par]))
