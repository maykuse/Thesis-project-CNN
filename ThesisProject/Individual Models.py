import numpy
import array
import torch.nn.init
import torchvision.transforms
from UNET import *
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 20
batch_size = 10
learning_rate = 0.001
total_epochs = 0

train_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_train')
validation_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_val')
test_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_test')
# order of the parameters are:
# t2m, u10, v10, z, t, tcc, tp, lsm, orog, slt

params = ["t2m", "u10", "v10", 'z', 't', "tcc", "tp"]
params_x = ["t2m", "v10", 'z', 't', "tcc", "tp"]
params_C = ["T2M", "U10", "V10", 'Z', 'T', "TCC", "TP"]

input_indices = [torch.tensor([1, 2, 3, 4, 5, 6]), torch.tensor([0, 2, 3, 4, 5, 6]), torch.tensor([0, 1, 3, 4, 5, 6]),
                 torch.tensor([0, 1, 2, 4, 5, 6]),
                 torch.tensor([0, 1, 2, 3, 5, 6]), torch.tensor([0, 1, 2, 3, 4, 6]), torch.tensor([0, 1, 2, 3, 4, 5])]

input_indices_x = [torch.tensor([2, 3, 4, 5, 6]), torch.tensor([0, 3, 4, 5, 6]), torch.tensor([0, 2, 4, 5, 6]),
                   torch.tensor([0, 2, 3, 5, 6]),  torch.tensor([0, 2, 3, 4, 6]), torch.tensor([0, 2, 3, 4, 5])]
out_index = [torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), torch.tensor([3]),
             torch.tensor([4]), torch.tensor([5]), torch.tensor([6])]

norm = transforms.Normalize((2.78415200e+02, -1.00402647e-01, 2.20140679e-01, 5.40906312e+04,
                             2.74440506e+02, 6.76697789e-01, 9.80986749e-05, 3.37078289e-01,
                             3.79497583e+02, 6.79204298e-01),
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


def gaussian_dropout(data: torch.Tensor, p: torch.Tensor):
    """
    Function for applying parametric Gaussian dropout
    Parameters:
        data: input data, expected shape (batch_size, num_channels, h, w)
        p: dropout rates in [0, 1], expected shape (batch_size, num_channels)
    Returns:
        out: Gaussian dropout output, shape (batch_size, num_channels, h, w)
    """
    p = p.view(*p.shape, 1, 1)
    alpha = p / (1. - p)
    noise = torch.randn_like(data)
    weights = 1. + torch.sqrt(alpha) * noise
    # this is weights = theta + theta * sqrt(alpha), with fixed theta = 1
    out = weights * data
    return out

def gaussian_dropout_image(data: torch.Tensor, p: torch.Tensor):
    """
        The only difference: p is given as 3D to keep track of the gradient of pixels
        p: dropout rates in [0, 1], expected shape (batch_size, num_channels, h, w)
    """
    # p = p.view(*p.shape, 1, 1)
    alpha = p / (1. - p)
    noise = torch.randn_like(data)
    weights = 1. + torch.sqrt(alpha) * noise
    # this is weights = theta + theta * sqrt(alpha), with fixed theta = 1
    out = weights * data
    return out


train_dataset = TrainDataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ValDataset()
val_loader = torch.utils.data.DataLoader(val_dataset)

test_dataset = TestDataset()
test_loader = torch.utils.data.DataLoader(test_dataset)

train_saved = False
all_individual_models = False

train_model = UNet(6, 1).to(device)
train_model.apply(init_weights)
loss_type = nn.L1Loss()
loss_type_nored = nn.L1Loss(reduction='none')
optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

if (train_saved):
    train_state = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/u10_nod_no_t2m')
    train_model.load_state_dict(train_state['state_dict'])
    optimizer.load_state_dict(train_state['optimizer'])
    total_epochs = train_state['epoch']

model = numpy.empty(7, dtype=UNet)
state = numpy.empty(7, dtype=dict)

if (all_individual_models):
    for i in range(7):
        model[i] = UNet(6, 1).to(device)

    for i, j in zip(range(7), params_x):
        state[i] = torch.load(
            '/home/ge75tis/Desktop/oezyurt/model//u10_no_{parameter}'.format(parameter=j))

    for i in range(7):
        model[i].load_state_dict(state[i]['state_dict'])

y_loss = {}
y_loss['train'] = []
y_loss['val'] = []
x_epoch = []

training = True


def train(epoch):
    train_model.train()
    train_loss = 0
    for i, (inputs) in enumerate(train_loader):
        input = inputs.index_select(dim=1, index=input_indices[0]).to(device, dtype=torch.float32)
        target = inputs.index_select(dim=1, index=out_index[0]).to(device, dtype=torch.float32)

        # Random amounts of(chosen from uniform d.) Gaussian Noise is added to the training inputs
        p = np.empty([len(input), 6])
        for j in range(len(input)):
            p[j] = np.random.uniform(low=np.nextafter(0, 1), high=np.nextafter(1, 0), size=6)
        tens_p = torch.tensor(p).to(device, dtype=torch.float32)
        noisy_input = gaussian_dropout(input, tens_p)

        optimizer.zero_grad()
        output = train_model(noisy_input)
        loss = loss_type(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(inputs)

    if (epoch >= 10 and epoch % 2 == 0):
        scheduler.step()

    avg_train_loss = train_loss / len(train_dataset)
    print(f'Epoch: {total_epochs}, Average Training Loss: {avg_train_loss:.6f}')
    # log_scalar("t2m_dropout_individual_train_loss", avg_train_loss)
    # y_loss['train'].append(avg_train_loss)


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
gradient_bars = True
draw_world_map = False
bar_chart_std = False
oneD_p = True
threeD_p = False
multi_parameter = False
draw_grad_bar = True

new_labels = [['u10', 'v10', 'z', 'tcc', 'tp'], ['t2m', 'z', 't', 'tcc', 'tp'],['t2m', 'z', 't', 'tcc', 'tp'],
              ['t2m', 'u10', 'v10', 'tcc', 'tp'], ['u10', 'v10', 'z', 'tcc', 'tp'], ['t2m', 'u10', 'z', 't', 'tp'], ['t2m', 'u10', 'z', 't', 'tcc']]

def val():
    counter = 0
    val_loss1 = 0
    index_month = 0
    if (all_individual_models):
        for i in range(6):
            model[i].eval()

    # grid_x = [0.00001, 0.1, 0.5, 0.999, 0.9999, 0.99995, 0.99999]
    # grid = [0.00001, 0.9, 0.999, 0.9999, 0.99995, 0.99999, 0.999999]
    # grid_x= [0.00001, 0.99999]
    # grid = [0.00001, 0.999999]

    np.set_printoptions(suppress=True)
    if(multi_parameter == False):
        for k in range(7):
            train_model = UNet(6, 1).to(device)
            labels = ["t2m", "u10", "v10", 'z', 't', "tcc", "tp"]
            train_state = torch.load(
                '/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/{par}_gaussiandropout_35_epochs'.format(par=labels[k]))
            train_model.load_state_dict(train_state['state_dict'])
            optimizer.load_state_dict(train_state['optimizer'])
            total_epochs = train_state['epoch']
            print(f'{labels[k]}')
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


            'You can change one channels noise to see if the gradients of others behaves differently.'


            for i, (vals) in enumerate(val_loader):

                'Choose one or more time points to look at gradient sensitivity maps'
                valid = vals.index_select(dim=1, index=input_indices[k]).to(device, dtype=torch.float32)
                target = vals.index_select(dim=1, index=out_index[k]).to(device, dtype=torch.float32)

                # we want gradient of 1,32,64 output wrt 6,32,64 noise p space
                if(threeD_p):
                    noisy_input = gaussian_dropout_image(valid, p)
                    output = train_model(noisy_input)
                    loss = loss_type(output, target)
                    loss = torch.squeeze(loss)
                if(oneD_p):
                    noisy_input = gaussian_dropout(valid, p)
                    output = train_model(noisy_input)
                    loss = loss_type(output, target)



                optimizer.zero_grad()
                'Choose the individual pixel from which the gradients will be computed'
                if(threeD_p):
                    loss.backward(inputs=p, retain_graph=True)
                if(oneD_p):
                    loss.backward(inputs=p, retain_graph=True)
                    for j in range(6):
                        std_tensor[j][i] = p_gradients[j]

                'Add gradients of input data. You can either take their average later or look at the sum'
                p_gradients.add_(p.grad)
                p.grad = None


            p_gradients.divide_(10000)
            # p_gradients.mul_(1000)
            # p_gradients = torch.abs(p_gradients)
            results = np.array(p_gradients.cpu())


            if (draw_world_map):
                fig = plt.figure(figsize=(10, 10))
                sns.set(font_scale=2.4)
                sns.heatmap(results[l], cmap="Reds", xticklabels=False, yticklabels=False,
                                cbar_kws=dict(use_gridspec=False, orientation="horizontal"))
                plt.title("\n".join(wrap("{param} Prediction Heatmap of Gradients wrt. p_{par} when p_all ~ 1".format(param=params_C[k], par=all_labels[k][l]), 35)))
                # plt.title("{param} Prediction model World heatmap of Gradients wrt. p_{par} when p_all ~ 1".format(param=params_C[k], par=all_labels[k][l]))
                plt.show()
                plt.tight_layout()
                fig.savefig('/home/ge75tis/Desktop/{param}_world_heatmap_{par}'.format(
                            param=params_C[k], par=all_labels[k][l]))

            if(draw_grad_bar):
                x = np.arange(len(all_labels[k]))

                width = 0.3
                fig3, ax = plt.subplots()
                std = torch.std(std_tensor, dim=1)
                # std = torch.mul(std, 1000)
                std = torch.div(std, 10000)
                print(results)
                # print(std)
                rects1 = ax.bar(x, results, width, yerr=std, capsize=4)
                ax.set_ylabel('Avg. Gradient of loss wrt. input parameters noise')
                ax.set_title('{param} Analysis of Gradient wrt. parameter noise when p_all ~ 1'.format(param=params_C[k]))
                ax.set_xticks(x, all_labels[k])
                # ax.bar_label(rects1, padding=3)
                fig3.tight_layout()
                plt.show()
                fig3.savefig('/home/ge75tis/Desktop/{param}_gradient_bar_chart'.format(
                        param=params_C[k]))



    set_to_zero = [3, 1, 1, 3, 0, 2, 2]
    params_zero = [4, 2, 1, 4, 0, 2, 2]

    if(multi_parameter):
        for k in range(7):
            train_model = UNet(6, 1).to(device)
            labels = ["t2m", "u10", "v10", 'z', 't', "tcc", "tp"]
            train_state = torch.load(
                '/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/{par}_gaussiandropout_35_epochs'.format(par=labels[k]))
            train_model.load_state_dict(train_state['state_dict'])
            optimizer.load_state_dict(train_state['optimizer'])
            total_epochs = train_state['epoch']

            print(f'{labels[k]}')
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
                    noisy_input = gaussian_dropout(valid, p)
                    noisy_input1 = gaussian_dropout(valid, p1)

                    output = train_model(noisy_input)
                    output1 = train_model(noisy_input1)

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

    if (all_individual_models):
        for i, (vals) in enumerate(val_loader):
            for j in range(6):
                valid[j] = vals.index_select(dim=1, index=input_indices_x[j]).to(device, dtype=torch.float32)
                target[j] = vals.index_select(dim=1, index=out_index[1]).to(device, dtype=torch.float32)

            for j in range(6):
                output[j] = model[j](valid[j])

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
    if(all_individual_models):
        for i in range(7):
            model[i].eval()
    for i, (tests) in enumerate(test_loader):
        test = tests.index_select(dim=1, index=input_indices[1]).to(device, dtype=torch.float32)
        target = tests.index_select(dim=1, index=out_index[1]).to(device, dtype=torch.float32)

        output = train_model(test)
        loss = loss_type(output, target)
        test_loss1 += loss.item()

    if (all_individual_models):
        for i, (tests) in enumerate(test_loader):
            for j in range(7):
                testy[j] = tests.index_select(dim=1, index=input_indices[j]).to(device, dtype=torch.float32)
                target[j] = tests.index_select(dim=1, index=out_index[j]).to(device, dtype=torch.float32)

            for j in range(7):
                output[j] = model[j](testy[j])

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
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('lr', learning_rate)
    mlflow.log_param('total_epochs', num_epochs)

    # val()
    # test()
    # for epoch in range(num_epochs):
    #     train(epoch)
    #     total_epochs += 1
    #     val()

    # val()
    # test()



save_model = False
if(save_model):
    train_state = {
        'epoch': num_epochs,
        'state_dict': train_model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(train_state, '/home/ge75tis/Desktop/oezyurt/model/5_1_UNET/u10_no_tp')

params_g = [['u10', 'v10', 'z', 'tcc', 'tp'], ['t2m', 'z', 't', 'tcc', 'tp'],['t2m', 'z', 't', 'tcc', 'tp'],
              ['t2m', 'u10', 't', 'tcc', 'tp'], ['u10', 'v10', 'z', 'tcc', 'tp'], ['t2m', 'u10', 'z', 't', 'tp'], ['t2m', 'u10', 'z', 't', 'tcc']]

bar_chart_std = False
params_zer = [4, 2, 1, 2, 0, 2, 2]
if(bar_chart_std):
    for i in range(7):
        grads = [[70, 65, 105, 41, 25], [26, 391, 232, 83, 282], [100, 341, 248, 181, 461],  [98, 227, 216, 50, 38],
                 [118, 115, 169, 64, 20], [37, 234, 116, 117, 241], [0.25, 31, 5, 0.15, 27]]
        clm_losses = [0.1061, 0.5411, 0.6394, 0.2035, 0.1897, 0.6936, 0.354]
        x = np.arange(5)
        width = 0.3
        plt.rcParams.update({'font.size': 25})
        fig, ax = plt.subplots(figsize=(11,8))
        # low = min(unet_losses)
        # high = max(unet_losses)
        # plt.ylim(0.24, 0.28)
        rects1 = ax.bar(x, grads[i], width, label="Gradient of {param} loss for p_{best} ~ 0 and p_others ~ 1".format(param=params_C[i], best=params_C[params_zer[i]]))
        # rects2 = ax.bar(x + width/2, clm_losses, width, label="Climatology")
        ax.set_ylabel('Average Gradient')
        ax.set_xticks(x, params_g[i])
        ax.set_xlabel('input parameters')
        # ax.set_title('UNet(5-to-1) prediction when one additional input is ignored')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
                  ncol=3, fancybox=True, shadow=True)

        # ax.bar_label(rects1, padding=3)
        # ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        plt.show()
        fig.savefig('/home/ge75tis/Desktop/higher_order_grad_{param}'.format(param=params_C[i]))


    # set_to_zeros = [4, 2, 1, 2, 0, 2, 2]
    # for i in range(7):
    #     loss_p = rec_losses[i][0]
    #     std_p = rec_losses[i][1]
    #     loss1_p = rec_losses1[i][0]
    #     std1_p = rec_losses1[i][1]
    #     x = np.arange(len(all_labels[i]))
    #     width = 0.3
    #     fig3, ax = plt.subplots()
    #     rects1 = ax.bar(x - width/2, loss_p, width, yerr=std_p, capsize=4, label='p_{aa} ~ 0 and p_y ~ 0, p_other ~ 1'.format(param=params_C[i]), aa=params[set_to_zeros[i]])
    #     # rects2 = ax.bar(x + width/2, loss1_p, width, yerr=std1_p, capsize=4, label='p_x ~ 1, p_other ~ 0'.format(param=params_C[i]))
    #     ax.set_ylabel('Average validation losses')
    #     ax.set_xlabel('parameter y')
    #     ax.set_title('{param} Dropout Analysis noise comparison when p_{aa} ~ 0'.format(param=params_C[i]), aa=params[set_to_zeros[i]])
    #     ax.set_xticks(x, all_labels[i])
    #     ax.legend()
    #     # ax.bar_label(rects1, padding=3)
    #     fig3.tight_layout()
    #     plt.show()
    #     fig3.savefig('/home/ge75tis/Desktop/double_dropout_bar_chart_with_stds_{param}'.format(param=params_C[i]))


distance_cluster = False
if(distance_cluster):
    comb_heatmap_percentage = [[0, 0.2372, 0.3441, 0.5068, 0.5326, 0.1715, 0.0742], [0.0130, 0, 0.6495, 0.4286, 0.1589, 0.1426, 0.2390],
                                   [0.0774, 0.7039, 0, 0.4368, 0.1069, 0.1861, 0.1784], [0.1943, 0.3980, 0.2435, 0, 0.5723, 0.0394, 0.0959],
                                   [0.7112, 0.0709, 0.1207, 0.5341, 0, 0.0857, 0.0063], [0.0351, 0.1879, 0.3602, 0.1391, 0.2310, 0, 0.1881],
                                   [0.0210, 0.4211, 0.4670, 0.0889, 0.0503, 0.2109, 0]]

    grad_heatmap = [[0, 37.1591, 53.27599, 23.071262, 369.16653, 10, 10],
                    [18.599178, 0, 506.08084, 239.3966, 252.6608, 99.09486, 122.54916],
                    [89.44, 419.50064, 0, 200.04277, 229.186, 160.98209, 173.02988],
                    [69.64928, 222.90337, 303.3208, 0, 192.48601, 38.59495, 34.078014],
                    [243.56503, 70.133736, 118.08318, 162.23822, 0, 48.33932, 16.240177],
                    [20.356546, 117.38441, 187.36948, 116.907524, 100.78966, 0, 154.77966],
                    [1, 10.919211, 24.24271, 3.0331063, 1, 16.8139, 0]]
    # how to deal with the really low gradients of tcc and tp? the distances become out of scale compared to others

    dist_matr = np.empty([7,7])
    dist_matr = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(comb_heatmap_percentage))
    # for i in range(7):
    #     for j in range(7):
    #         if(i == j):
    #             dist_matr[i][j] = 0
    #         else:
    #             dist_matr[i][j] *= (1 / (grad_heatmap[i][j] + grad_heatmap[j][i]) ) * 1000

    print(dist_matr)

    dt = [('len', float)]
    dist_matr = dist_matr.view(dt)

    G = nx.from_numpy_matrix(dist_matr)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))
    G = nx.drawing.nx_agraph.to_agraph(G)
    G.node_attr.update(color="red", style="filled")
    G.edge_attr.update(color="white", width="0.0")
    fig = plt.figure()
    G.draw('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/a_cluster2.png', format='png', prog='neato')
    # plt.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/cluster2.png', format="PNG")




heatmap = True
if(heatmap):
    new_param_order = ['t', 't2m', 'z', 'v10', 'u10', 'tp', 'tcc']
    # fig1 = plt.figure(figsize=(14,20))

    comb_heatmap_percentage = [[1, 0.2372, 0.3441, 0.5068, 0.5326, 0.1715, 0.0742], [0.0130, 1, 0.6495, 0.4286, 0.1589, 0.1426, 0.2390],
                               [0.0774, 0.7039, 1, 0.4368, 0.1069, 0.1861, 0.1784], [0.1943, 0.3980, 0.2435, 1, 0.5723, 0.0394, 0.0959],
                               [0.7112, 0.0709, 0.1207, 0.5341, 1, 0.0857, 0.0063], [0.0351, 0.1879, 0.3602, 0.1391, 0.2310, 1, 0.1881],
                               [0.0210, 0.4211, 0.4670, 0.0889, 0.0503, 0.2109, 1]]
    comb_heatmap = [[0, 4.30, 3.65, 2.77, 2.60, 4.65, 5.20], [11.23, 0, 3.94, 6.46, 9.55, 9.75, 8.53],
                    [16.08, 5.06, 0, 9.83, 15.68, 14.13, 14.36],
                    [10.56, 7.81, 9.87, 0, 5.57, 12.53, 11.78], [2.75, 8.91, 8.42, 4.52, 0, 8.77, 9.50],
                    [4.77, 4.06, 3.15, 4.23, 3.77, 0, 3.98], [1.09, 0.65, 0.59, 1.00, 1.05, 0.87, 0]]
    #, -18.684557, -9.69056 ]
    grad_heatmap = [[0, 37.1591, 53.27599, 23.071262, 369.16653, -18.684557, -9.69056], [18.599178, 0, 506.08084, 239.3966, 252.6608, 99.09486, 122.54916 ],
                    [89.44, 419.50064, 0, 200.04277, 229.186, 160.98209, 173.02988], [69.64928, 222.90337, 303.3208, 0, 192.48601, 38.59495, 34.078014],
                    [243.56503, 70.133736, 118.08318, 162.23822, 0, 48.33932, 16.240177], [20.356546, 117.38441, 187.36948, 116.907524, 100.78966, 0, 154.77966],
                    [0.11764815, 10.919211, 24.24271, 3.0331063, 0.18807021, 16.8139, 0]]

    grad_heatmap2 = [[0, 37.1591, 53.27599, 23.071262, 369.16653, 1, 1], [18.599178, 0, 506.08084, 239.3966, 252.6608, 99.09486, 122.54916 ],
                    [89.44, 419.50064, 0, 200.04277, 229.186, 160.98209, 173.02988], [69.64928, 222.90337, 303.3208, 0, 192.48601, 38.59495, 34.078014],
                    [243.56503, 70.133736, 118.08318, 162.23822, 0, 48.33932, 16.240177], [20.356546, 117.38441, 187.36948, 116.907524, 100.78966, 0, 154.77966],
                    [0.11764815, 10.919211, 24.24271, 3.0331063, 0.18807021, 16.8139, 0]]

    grad_heatmap_new = [[0, 0.36795822, 0.43980718, 0.61094254, 0.66767836, 0.25484586, 0.15239869], [0.09645872, 0, 3.8159883,  1.0963811, 1.3558255,  0.1850218, 0.49045599],
                        [0.97554576, 6.065276,  0, 2.6048186,  1.7355888,  1.7813991,  0.6647806, ], [1.5544034, 3.0572004, 3.1909885, 0, 4.1371703, 0.4340603, 0.7173816],
                        [4.782727,   0.28787705, 0.8700743,  2.5207248, 0, 0.62985164, 0.08375259], [0.1033018,  0.3755193, 0.7591337,  0.58453566, 0.48831362, 0, 0.65765136],
                        [0.00877767, 0.09491006, 0.12384017, 0.04167771,  0.02347502, 0.07231351, 0]]

    testa = numpy.array(grad_heatmap_new)
    print(testa.mean(axis=1))
    print(testa.std(axis=1))
    testa1 = torch.from_numpy(testa)

    norm_testy = transforms.Normalize((0.35623298, 1.00573306, 1.97534412, 1.87017207, 1.31071534, 0.42406506,
    0.05214202), (0.22287954, 1.24244417, 1.84538173, 1.47752059, 1.62332719, 0.26234905,
    0.04294045))

    test3 = norm_testy(testa1)
    print(test3)



    # grad_heatmap_new =

    heat_norm = plt.Normalize(0,1)
    dist_matr = np.empty([7, 7])
    for i in range(7):
        for j in range(7):
            if (i == j):
                dist_matr[i][j] = 0
            else:
                dist_matr[i][j] = (1 / (grad_heatmap_new[i][j] + grad_heatmap_new[j][i]))

    # print(dist_matr)

    sns.set(font_scale=1.2)
    # sns.heatmap(grad_heatmap, linewidths=.5, cmap="magma", annot=True, xticklabels=params_C, yticklabels=params_C, norm=LogNorm(), fmt='.3g')
    fig1 = sns.clustermap(comb_heatmap_percentage, cbar_kws={"shrink": 0.5}, method='average', linewidths=1, linecolor='white', row_linkage=scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dist_matr)), col_linkage=scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dist_matr)), cmap="magma",  annot=True, xticklabels=params, yticklabels=params, norm=heat_norm, fmt='.3g')
    fig1.ax_heatmap.set_xticklabels(fig1.ax_heatmap.get_xmajorticklabels(), fontsize = 22)
    fig1.ax_heatmap.set_yticklabels(fig1.ax_heatmap.get_ymajorticklabels(), fontsize = 22)

    # Why does the clustermap change tcc and tp's position, even though they should lie at the end
    # plt.title('Gradient of prediction loss wrt. input parameters when p_all ~ 1')
    # plt.xlabel('input parameters (gradient)')
    # plt.ylabel('predicted parameter')
    # plt.title('Gradient of prediction loss with respect to p of each parameter when p_all ~ 1', loc='center', wrap=True)
    # plt.xlabel('input parameters')
    # plt.ylabel('predicted parameter')
    fig = plt.figure()
    plt.show()
    # fig1.savefig('/home/ge75tis/Desktop/trueHeatmap')


    dist = scipy.spatial.distance.squareform(dist_matr)
    links = scipy.cluster.hierarchy.linkage(dist, "average")
    scipy.cluster.hierarchy.dendrogram(links, labels=params_C)

    plt.title("Hierarchical Clustering of Parameters using Gradients (avg.)")
    plt.ylabel("distance")
    plt.show()

    plt.savefig('/home/ge75tis/Desktop/newgrad')

# row_linkage=scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(dist_matr))