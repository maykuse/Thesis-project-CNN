import numpy
import array
import torch.nn.init
import torchvision.transforms

import UNET_parts
from UNET import *
from PrepareDataset import *
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
    help="Set to true just to see test results on already trained model"
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
parser.add_argument(
    "--all-individual-models",
    type=bool,
    default=False,
    metavar="Bool",
    help="Set True to load all individual models at the same time"
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_epochs = 0


params = ["t2m", "u10", "v10", 'z', 't', "tcc", "tp"]
# params_x = ["t2m", "v10", 'z', 't', "tcc", "tp"]
params_C = ["T2M", "U10", "V10", 'Z', 'T', "TCC", "TP"]

# THINK ABOUT HOW TO MAKE THIS MORE COMPACT and also how to create the input_indices_x implicitly
input_indices = [torch.tensor([1, 2, 3, 4, 5, 6]), torch.tensor([0, 2, 3, 4, 5, 6]), torch.tensor([0, 1, 3, 4, 5, 6]),
                 torch.tensor([0, 1, 2, 4, 5, 6]),
                 torch.tensor([0, 1, 2, 3, 5, 6]), torch.tensor([0, 1, 2, 3, 4, 6]), torch.tensor([0, 1, 2, 3, 4, 5])]
# input_indices_x = [torch.tensor([2, 3, 4, 5, 6]), torch.tensor([0, 3, 4, 5, 6]), torch.tensor([0, 2, 4, 5, 6]),
#                    torch.tensor([0, 2, 3, 5, 6]),  torch.tensor([0, 2, 3, 4, 6]), torch.tensor([0, 2, 3, 4, 5])]
out_index = [torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), torch.tensor([3]),
             torch.tensor([4]), torch.tensor([5]), torch.tensor([6])]


train_dataset = PrepareDataset.TrainDataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

val_dataset = PrepareDataset.ValDataset()
val_loader = torch.utils.data.DataLoader(val_dataset)

test_dataset = PrepareDataset.TestDataset()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size)

train_saved = False


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)

model = UNet(6, 1).to(device)
model.apply(init_weights)
loss_type = nn.L1Loss()
loss_type_nored = nn.L1Loss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

if (args.load_model):
    train_state = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/u10_nod_no_t2m')
    model.load_state_dict(train_state['state_dict'])
    optimizer.load_state_dict(train_state['optimizer'])
    total_epochs = train_state['epoch']

models = numpy.empty(7, dtype=UNet)
state = numpy.empty(7, dtype=dict)

if (args.all_individual_models):
    for i in range(7):
        models[i] = UNet(6, 1).to(device)

    for i, j in zip(range(7), params):
        state[i] = torch.load(
            '/home/ge75tis/Desktop/oezyurt/model/6_1_UNET_TRUE/{parameter}'.format(parameter=j))

    for i in range(7):
        models[i].load_state_dict(state[i]['state_dict'])

y_loss = {}
y_loss['train'] = []
y_loss['val'] = []
x_epoch = []


def train(epoch):
    model.train()
    train_loss = 0
    for i, (inputs) in enumerate(train_loader):
        input = inputs.index_select(dim=1, index=input_indices[args.output_parameter]).to(device, dtype=torch.float32)
        target = inputs.index_select(dim=1, index=out_index[args.output_parameter]).to(device, dtype=torch.float32)

        # Random amounts of(chosen from uniform d.) Gaussian Noise is added to the training inputs
        p = np.empty([len(input), 6])
        for j in range(len(input)):
            p[j] = np.random.uniform(low=np.nextafter(0, 1), high=np.nextafter(1, 0), size=6)
        tens_p = torch.tensor(p).to(device, dtype=torch.float32)
        noisy_input = UNET_parts.gaussian_dropout(input, tens_p)

        optimizer.zero_grad()
        output = model(noisy_input)
        loss = loss_type(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(inputs)

    if (epoch >= 10 and epoch % 2 == 0):
        scheduler.step()

    avg_train_loss = train_loss / len(train_dataset)
    print(f'Epoch: {total_epochs}, Average Training Loss: {avg_train_loss:.6f}')
    log_scalar("{par}_dropout_train_loss".format(par=params[args.output_parameter]), avg_train_loss)
    y_loss['train'].append(avg_train_loss)



valid = numpy.empty(7, dtype=torch.Tensor)
testy = numpy.empty(7, dtype=torch.Tensor)
target = numpy.empty(7, dtype=torch.Tensor)
output = numpy.empty(7, dtype=torch.Tensor)
loss = numpy.empty(7, dtype=torch.Tensor)
val_loss = [0 for i in range(7)]
test_loss = [0 for i in range(7)]
avg_val_loss = [0 for i in range(7)]
avg_test_loss = [0 for i in range(7)]
avg_val_loss_gridded_m = [[[0 for l in range(12)] for k in range(6)] for j in range(7)]
avg_val_loss_gridded = [[0 for k in range(6)] for j in range(7)]


month = False
graph = False
num_of_days_monthly = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
inc_point = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
index_month = 0

all_labels = [["u10", "v10", 'z', 't', "tcc", "tp"], ["t2m", "v10", 'z', 't', "tcc", "tp"],
              ["t2m", "u10", 'z', 't', "tcc", "tp"], ["t2m", "u10", "v10", 't', "tcc", "tp"],
              ["t2m", "u10", "v10", 'z', "tcc", "tp"], ["t2m", "u10", "v10", 'z', 't', "tp"],
              ["t2m", "u10", "v10", 'z', 't', "tcc"]]

rec_losses = torch.zeros([7, 2, 6])
rec_losses1 = torch.zeros([7, 2, 6])
std_tensor = torch.zeros([6, 730])
std_tensor1 = torch.zeros(730)

save_figs = False
gradient_bars = False
draw_world_map = True
bar_chart_std = False
oneD_p = True
threeD_p = False
multi_parameter = False
draw_grad_bar = False

new_labels = [['u10', 'v10', 'z', 'tcc', 'tp'], ['t2m', 'z', 't', 'tcc', 'tp'],['t2m', 'z', 't', 'tcc', 'tp'],
              ['t2m', 'u10', 'v10', 'tcc', 'tp'], ['u10', 'v10', 'z', 'tcc', 'tp'], ['t2m', 'u10', 'z', 't', 'tp'], ['t2m', 'u10', 'z', 't', 'tcc']]





def val():
    counter = 0
    val_loss1 = 0
    index_month = 0
    if (args.all_individual_models):
        for i in range(6):
            models[i].eval()

    # grid_x = [0.00001, 0.1, 0.5, 0.999, 0.9999, 0.99995, 0.99999]
    # grid = [0.00001, 0.9, 0.999, 0.9999, 0.99995, 0.99999, 0.999999]

    np.set_printoptions(suppress=True)
    set_to_zero = [3, 1, 1, 3, 0, 2, 2]
    params_zero = [4, 2, 1, 4, 0, 2, 2]
    if(multi_parameter == False):
        for k in range(7):
            lrp_map = torch.zeros([1, 6, 32, 64]).to(device, dtype=torch.float32)
            model = UNet(6, 1).to(device)
            train_state = torch.load(
                '/home/ge75tis/Desktop/oezyurt/model/6_1_UNET_TRUE/{par}'.format(par=params[k]))
            model.load_state_dict(train_state['state_dict'])
            optimizer.load_state_dict(train_state['optimizer'])
            total_epochs = train_state['epoch']
            print(f'{params[k]}')
            #for l in range(6):
            # either create a 1D p vector (only tracks loss), or a 3D p vector (tracks pixel gradients)
            if(threeD_p):
                p_gradients = torch.zeros([6,32,64]).to(device, dtype=torch.float32)
                p = torch.zeros([6, 32, 64], requires_grad=True).to(
                    device, dtype=torch.float32)
                'Fill any value you want. Closer values to 1 means higher gradients.'
                p.fill_(0.75)

            if(oneD_p):
                p_gradients = torch.zeros(6).to(device, dtype=torch.float32)
                p = torch.zeros(6, requires_grad=True).to(device, dtype=torch.float32)
                p.fill_(0.999)
                p[set_to_zero[k]] = 0.00001


            'You can change one channels noise to see if the gradients of others behaves differently.'

            for i, (vals) in enumerate(val_loader):
                'Choose one or more time points to look at gradient sensitivity maps'
                if(i == 180):
                    valid = vals.index_select(dim=1, index=input_indices[k]).to(device, dtype=torch.float32)
                    target = vals.index_select(dim=1, index=out_index[k]).to(device, dtype=torch.float32)

                    # we want gradient of 1,32,64 output wrt 6,32,64 noise p space
                    if(threeD_p):
                        noisy_input = UNET_parts.gaussian_dropout_image(valid, p)
                        output = model(noisy_input)
                        loss = loss_type(output, target)
                        loss = torch.squeeze(loss)
                    if(oneD_p):
                        noisy_input = UNET_parts.gaussian_dropout(valid, p)
                        output = model(noisy_input)
                        loss = loss_type(output, target)
                        #model.rule = EpsilonRule()
                        lrp = captum.attr.LRP(model)
                        attribution = lrp.attribute(valid, target=(0, 16, 32))
                        lrp_map.add_(attribution)

                    optimizer.zero_grad()

            # lrp_map.divide_(len(val_dataset)
            lrp_map.mul_(100000)
            print(lrp_map)
                # 'Choose the individual pixel from which the gradients will be computed'
                # if(threeD_p):
                #     loss.backward(inputs=p, retain_graph=True)
                # if(oneD_p):
                #     loss.backward(inputs=p, retain_graph=True)
                #     for j in range(6):
                #         std_tensor[j][i] = p_gradients[j]
                #
                # 'Add gradients of input data. You can either take their average later or look at the sum'
                # p_gradients.add_(p.grad)
                # p.grad = None


            # # p_gradients.divide_(1000)
            #
            # p_gradients.mul_(1000)
            # # if(k == 5 or k == 6):
            # #     p_gradients.divide_(100)
            #
            # # p_gradients = torch.abs(p_gradients)
            results = np.array(lrp_map.detach().cpu().clone().numpy())
            #
            # v_maxes = [7.5, 10, 10, 5, 5, 7.5, 3]
            # v_mins = [-7.5, -10, -10, -5, -5, -7.5, -3]
            # v_maxes_pixel = [7.5, 100, 100, 15, 7.5, 2.5, 2.5]
            # v_mins_pixel = [-7.5, -100, -100, -15, -7.5, -2.5, -2.5]
            # abs_max = [10, 10, 10, 5, 5, 7.5, 4]
            # time_maxes = [40, 50, 50, 50, 25, 50, 20]
            # time_mins = [-40, -50, -50, -50, -25, -50, -20]
            # # vmin=v_mins[0], vmax=v_maxes[0],
            lrp_min = [-0.25, -1.5, -1.5, -0.15, -0.15, -10, -5]
            lrp_max = [40, 400, 250, 40, 30, 1500, 500]

            if (draw_world_map):
                for l in range(6):
                    fig = plt.figure(figsize=(10, 10))
                    sns.set(font_scale=2.2)
                    sns.heatmap(results[0][l], cmap="RdBu", xticklabels=False, yticklabels=False, center=0.00, vmin=-lrp_max[k], vmax=lrp_max[k],
                                    cbar_kws=dict(use_gridspec=False, orientation="horizontal"))
                    plt.title("\n".join(wrap("{param} Prediction Heatmap LRP with respect to input parameter {par}".format(param=params_C[k], par=all_labels[k][l]), 35)))
                    # plt.title("{param} Prediction model World heatmap of Gradients wrt. p_{par} when p_all ~ 1".format(param=params_C[k], par=all_labels[k][l]))
                    plt.show()
                    plt.tight_layout()
                    fig.savefig('/home/ge75tis/Desktop/LRP180/{param}_world_heatmap_{par}'.format(param=params_C[k], par=all_labels[k][l]))

            if(draw_grad_bar):
                x = np.arange(len(all_labels[k]))

                width = 0.3
                fig3, ax = plt.subplots()
                std = torch.std(std_tensor, dim=1)
                # std = torch.mul(std, 1000)
                std = torch.div(std, 10000)
                print(results)
                print(std)

                # print(std)
                rects1 = ax.bar(x, results, width, yerr=std, capsize=4)
                ax.set_ylabel('Avg. Gradient over Validation data')
                ax.set_xlabel('parameter x')
                ax.set_title('{param} Loss Gradients w.r.t. p_{par} ~ 0 and p_x ~ 0 when p_others ~ 1'.format(param=params_C[k], par=params[set_to_zero[k]]), fontsize=16)
                ax.set_xticks(x, all_labels[k])
                ax.tick_params(axis='x', which='major', labelsize=16)
                # ax.bar_label(rects1, padding=3)
                fig3.tight_layout()
                plt.show()
                # fig3.savefig('/home/ge75tis/Desktop/{param}_gradient_bar_chart'.format(param=params_C[k]))





    if(multi_parameter):
        for k in range(7):
            model = UNet(6, 1).to(device)
            train_state = torch.load(
                '/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/{par}_gaussiandropout_35_epochs'.format(par=params[k]))
            model.load_state_dict(train_state['state_dict'])
            optimizer.load_state_dict(train_state['optimizer'])
            total_epochs = train_state['epoch']

            print(f'{params[k]}')
            val_loss11 = 0
            for l in range(6): # Set each parameter noise to 0 one by one and look at the gradient of others at the same time
                p_gradients = torch.zeros(6).to(device, dtype=torch.float32)
                p = torch.tensor([0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999], requires_grad=True).to(
                    device, dtype=torch.float32)
                p[set_to_zero[k]] = 0.00001
                p1 = torch.tensor([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001], requires_grad=True).to(device, dtype=torch.float32)
                p[l] = 0.00001
                p1[l] = 0.999999

                for i, (vals) in enumerate(val_loader):
                    valid = vals.index_select(dim=1, index=input_indices[k]).to(device, dtype=torch.float32)
                    target = vals.index_select(dim=1, index=out_index[k]).to(device, dtype=torch.float32)

                    # tens_p = torch.tensor(p).to(device, dtype=torch.float32)
                    noisy_input = UNET_parts.gaussian_dropout(valid, p)
                    noisy_input1 = UNET_parts.gaussian_dropout(valid, p1)

                    output = model(noisy_input)
                    output1 = model(noisy_input1)

                    loss = loss_type(output, target)
                    loss1 = loss_type(output1, target)

                    val_loss1 += loss.item()
                    val_loss11 += loss1.item()

                    std_tensor[i] = loss.item()
                    std_tensor1[i] = loss1.item()

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
                    rec_losses1[k][0][l] = val_loss11 / len(val_dataset)
                    val_loss11 = 0
                    rec_losses1[k][1][l] = torch.std(std_tensor1)


                print(f'Average Validation Losses: {val_loss1/len(val_dataset):.6f}')
                avg_val_loss_gridded[k][l] = val_loss1/len(val_dataset)
                val_loss1 = 0


                if(month):
                    for i in range(12):
                        if (i == 0 or i == 2 or i == 4 or i == 6 or i == 7 or i == 9 or i == 11):
                            avg_val_loss_gridded_m[k][l][i] = avg_val_loss_gridded_m[k][l][i] / 62
                        elif (i == 3 or i == 5 or i == 8 or i == 10):
                            avg_val_loss_gridded_m[k][l][i] = avg_val_loss_gridded_m[k][l][i] / 60
                        elif (i == 1):
                            avg_val_loss_gridded_m[k][l][i] = avg_val_loss_gridded_m[k][l][i] / 56


                if(graph):
                    if(month):
                        x_axis = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
                        labels = all_labels[k]
                        fig = plt.figure()
                        for i in range(6):
                            plt.errorbar(avg_val_loss_gridded_m[k][i], yerr=0, label=labels[i])
                        fig.suptitle('{param} analysis, p_x ~ 0, p_other ~ 1 per month'.format(param=params_C[k]))
                        fig.errorbar
                        plt.legend()
                        plt.xlabel('months')
                        plt.ylabel('Average loss')
                        fig.savefig("/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/per_month_test/{param}_Dropout_per_month_p_x_0".format(param=params_C[k]))
                    # fig = plt.figure(g, figsize=(10,10))
                    # sns.heatmap(avg_val_loss_gridded[g], linewidths=.5, cmap="Greens", annot=True, xticklabels=grid, yticklabels=grid, norm=LogNorm(), fmt=".3f")
                    # labels = ["t2m", "u10", "v10", 'z', 't', "tcc"]
                    # plt.title('Avg Validation loss of TP for different dropout rates of {par} and other parameters'.format(par=labels[g]))
                    # plt.xlabel('other parameters dropout rate p')
                    # plt.ylabel('{par} dropout rate p'.format(par=labels[g]))
                    # fig.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/DROPOUT_ANALYSIS/tp_analysis_{label}_heatmap'.format(label=labels[g]))

                # results = np.array(p_gradients.cpu())
            results = rec_losses[k][0]

            if (gradient_bars):
                # print(f' {labels[k]}: p_grads, {all_labels[k]} p ~ 0, others ~ 1: {results}')
                x = np.arange(len(all_labels[k]))
                width = 0.3
                fig3, ax = plt.subplots()
                rects1 = ax.bar(x, results, width, label='p_{param} ~ 0 and p_x ~ 0, p_others ~ 1')
                ax.set_ylabel('Avg. Validation Loss')
                ax.set_xlabel
                ax.set_title(
                    '{param} Dropout Analysis when p_{par} ~ 0 and p_x ~ 0, p_others ~ 1'.format(param=params_C[k], par=params[params_zero[k]]))
                ax.set_xticks(x, all_labels[K])
                ax.legend()
                # ax.bar_label(rects1, padding=3)
                fig3.tight_layout()
                fig3.savefig(
                    '/home/ge75tis/Desktop/{param}_dropout_bar_chart'.format(
                        param=params_C[k]))

    if (args.all_individual_models):
        for i, (vals) in enumerate(val_loader):
            for j in range(6):
                valid[j] = vals.index_select(dim=1, index=input_indices_x[j]).to(device, dtype=torch.float32)
                target[j] = vals.index_select(dim=1, index=out_index[1]).to(device, dtype=torch.float32)

            for j in range(6):
                output[j] = models[j](valid[j])

            for j in range(6):
                loss[j] = loss_type(output[j], target[j])

            for j in range(6):
                val_loss[j] += loss[j].item()

            counter += 1

        for j in range(6):
            avg_val_loss[j] = val_loss[j] / len(val_dataset)
        print(f'Average Validation Losses: {avg_val_loss}')


def test():
    test_loss1 = 0
    if(args.all_individual_models):
        for i in range(7):
            models[i].eval()
    for i, (tests) in enumerate(test_loader):
        test = tests.index_select(dim=1, index=input_indices[1]).to(device, dtype=torch.float32)
        target = tests.index_select(dim=1, index=out_index[1]).to(device, dtype=torch.float32)

        output = model(test)
        loss = loss_type(output, target)
        test_loss1 += loss.item()

    if (args.all_individual_models):
        for i, (tests) in enumerate(test_loader):
            for j in range(7):
                testy[j] = tests.index_select(dim=1, index=input_indices[j]).to(device, dtype=torch.float32)
                target[j] = tests.index_select(dim=1, index=out_index[j]).to(device, dtype=torch.float32)

            for j in range(7):
                output[j] = models[j](testy[j])

            for j in range(7):
                loss[j] = loss_type(output[j], target[j])

            for j in range(7):
                test_loss[j] += loss[j].item()

        for j in range(7):
            avg_test_loss[j] = test_loss[j] / len(test_dataset)

        print(f'Average Test Losses: {avg_test_loss}')
        # log_scalar("a_individual_test_loss", avg_test_loss)


def log_scalar(name, value):
    mlflow.log_metric(name, value)


with mlflow.start_run() as run:
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('lr', learning_rate)
    mlflow.log_param('total_epochs', num_epochs)

    if(args.test == False):
        for epoch in range(args.epochs):
            train()
            val()
            total_epochs += 1

    print("Training finished. Final values:")
    val()
    test()



if(args.save_trained_model):
    train_state = {
        'epoch': num_epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(train_state, '/home/ge75tis/Desktop/oezyurt/model/5_1_UNET/u10_no_tp')
