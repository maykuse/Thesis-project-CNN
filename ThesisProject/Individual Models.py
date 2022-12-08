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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 35
batch_size = 10
learning_rate = 0.0005
total_epochs = 0

train_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_train')
validation_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_val')
test_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_test')
# order of the parameters are: t2m, u10, v10, z, t, tcc, tp, lsm, orog, slt

params = ["t2m", "u10", "v10", 'z', 't', "tcc", "tp"]

indices_no_wind = torch.tensor([0,3,4,5,6])
indices_no_tcc_tp = torch.tensor([0,1,2,3,4])

input_indices = [torch.tensor([1,2,3,4,5,6]), torch.tensor([0,2,3,4,5,6]), torch.tensor([0,1,3,4,5,6]), torch.tensor([0,1,2,4,5,6]),
                 torch.tensor([0,1,2,3,5,6]), torch.tensor([0,1,2,3,4,6]), torch.tensor([0,1,2,3,4,5])]
out_index = [torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), torch.tensor([3]), torch.tensor([4]), torch.tensor([5]), torch.tensor([6])]


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
    weights = 1. + torch.sqrt(alpha) * noise  # this is weights = theta + theta * sqrt(alpha), with fixed theta = 1
    out = weights * data
    return out


train_dataset = TrainDataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset =  ValDataset()
val_loader = torch.utils.data.DataLoader(val_dataset)

test_dataset = TestDataset()
test_loader = torch.utils.data.DataLoader(test_dataset)

train_saved = False
all_individual_models = False

train_model = UNet(6,1).to(device)
train_model.apply(init_weights)
loss_type = nn.L1Loss()
optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

if(train_saved):
    train_state = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/t2m_gaussiandropout_35_epochs')
    train_model.load_state_dict(train_state['state_dict'])
    optimizer.load_state_dict(train_state['optimizer'])
    total_epochs = train_state['epoch']

model = numpy.empty(7, dtype=UNet)
state = numpy.empty(7, dtype=dict)

if(all_individual_models):
    for i in range(7):
        model[i] = UNet(6,1).to(device)

    for i,j in zip(range(7), params):
        state[i] = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/{parameter}_gaussiandropout_35_epochs'.format(parameter=j))

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
        input = inputs.index_select(dim=1, index=input_indices[6]).to(device, dtype=torch.float32)
        target = inputs.index_select(dim=1, index=out_index[6]).to(device, dtype=torch.float32)

        # Gaussian Noise is added to the training inputs
        p = np.empty([len(input), 6])
        for j in range(len(input)):
            p[j] = np.random.uniform(low=np.nextafter(0,1), high=np.nextafter(1,0), size=6)
        tens_p = torch.tensor(p).to(device, dtype=torch.float32)
        noisy_input = gaussian_dropout(input, tens_p)

        optimizer.zero_grad()
        output = train_model(noisy_input)
        loss = loss_type(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(inputs)

    if(epoch >= 25 and epoch % 5 == 0):
        scheduler.step()

    avg_train_loss = train_loss/len(train_dataset)
    print(f'Epoch: {total_epochs}, Average Training Loss: {avg_train_loss:.6f}')
    log_scalar("tp_dropout_individual_train_loss", avg_train_loss)
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
avg_val_loss_gridded_m = [[[[0 for l in range(12)] for i in range(2)] for j in range(2)] for k in range(6)]
avg_val_loss_gridded = [[[0 for i in range(2)] for j in range(2)] for k in range(6)]

month = False
graph = False
num_of_days_monthly = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
inc_point = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
index_month = 0

all_labels = [["u10", "v10", 'z', 't', "tcc", "tp"], ["t2m", "v10", 'z', 't', "tcc", "tp"],["t2m", "u10", 'z', 't', "tcc", "tp"]
              ,["t2m", "u10", "v10", 't', "tcc", "tp"], ["t2m", "u10", "v10", 'z', "tcc", "tp"], ["t2m", "u10", "v10", 'z', 't', "tp"], ["t2m", "u10", "v10", 'z', 't', "tcc"]]

def val():
    counter = 0
    val_loss1 = 0
    if(all_individual_models):
        for i in range(7):
            model[i].eval()

    # grid_x = [0.00001, 0.1, 0.5, 0.999, 0.9999, 0.99995, 0.99999]
    # grid = [0.00001, 0.9, 0.999, 0.9999, 0.99995, 0.99999, 0.999999]
    # grid_x= [0.00001, 0.99999]
    # grid = [0.00001, 0.999999]

    np.set_printoptions(suppress=True)
    for k in range(7):
        train_model = UNet(6, 1).to(device)
        labels = ["t2m", "u10", "v10", 'z', 't', "tcc", "tp"]
        train_state = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/{par}_gaussiandropout_35_epochs'.format(par=labels[k]))
        train_model.load_state_dict(train_state['state_dict'])
        optimizer.load_state_dict(train_state['optimizer'])
        total_epochs = train_state['epoch']
        for l in range(6):
            p_gradients = torch.tensor([0, 0, 0, 0, 0, 0]).to(device, dtype=torch.float32)
            p = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01, 0.01], requires_grad=True).to(device, dtype=torch.float32)
            p[l] = 0.99

            for i, (vals) in enumerate(val_loader):
                valid = vals.index_select(dim=1, index=input_indices[k]).to(device, dtype=torch.float32)
                target = vals.index_select(dim=1, index=out_index[k]).to(device, dtype=torch.float32)

                # tens_p = torch.tensor(p).to(device, dtype=torch.float32)
                noisy_input = gaussian_dropout(valid, p)
                output = train_model(noisy_input)
                loss = loss_type(output, target)

                optimizer.zero_grad()
                loss.backward(inputs=p, retain_graph=True)
                for j in range(6):
                    p_gradients[j] += p.grad[j]

                val_loss1 += loss.item()

            for i in range(6):
                p_gradients[i] = p_gradients[i] / len(val_dataset)

            results = np.array(p_gradients.cpu())

            print(f' {labels[k]}: p_grads, {all_labels[k][l]} p ~ 1, others ~ 0: {results}')


    # for g in range(6):
    #     for k in range(2):
    #         for l in range(2):
    #             if(k == l):
    #                 continue
    #             for i, (vals) in enumerate(val_loader):
    #                 valid = vals.index_select(dim=1, index=input_indices[6]).to(device, dtype=torch.float32)
    #                 target = vals.index_select(dim=1, index=out_index[6]).to(device, dtype=torch.float32)
    #
    #                 p = np.empty([len(valid), 6])
    #                 if(g == 0):
    #                     p[0] = [grid_x[k], grid[l], grid[l], grid[l], grid[l], grid[l]]
    #                 if(g == 1):
    #                     p[0] = [grid[l], grid_x[k], grid[l], grid[l], grid[l], grid[l]]
    #                 if(g == 2):
    #                     p[0] = [grid[l], grid[l], grid_x[k], grid[l], grid[l], grid[l]]
    #                 if(g == 3):
    #                     p[0] = [grid[l], grid[l], grid[l], grid_x[k], grid[l], grid[l]]
    #                 if(g == 4):
    #                     p[0] = [grid[l], grid[l], grid[l], grid[l], grid_x[k], grid[l]]
    #                 if(g == 5):
    #                     p[0] = [grid[l], grid[l], grid[l], grid[l], grid[l], grid_x[k]]
    #
    #                 tens_p = torch.tensor(p).to(device, dtype=torch.float32)
    #                 noisy_input = gaussian_dropout(valid, tens_p)
    #                 output = train_model(noisy_input)
    #                 loss = loss_type(output, target)
    #                 val_loss1 += loss.item()
    #
    #
    #                 if (month):
    #                     if (counter in inc_point):
    #                         index_month += 1
    #                     if (counter == 365):
    #                         counter = 0
    #                         index_month = 0
    #                     avg_val_loss_gridded_m[g][k][l][index_month] += loss.item()
    #
    #                 counter += 1
    #
    #             print(f'Average Validation Losses: {val_loss1/len(val_dataset):.6f}')
    #             avg_val_loss_gridded[g][k][l] = val_loss1/len(val_dataset)
    #             val_loss1 = 0
    #
    #
    #             if(month):
    #                 for i in range(12):
    #                     if (i == 0 or i == 2 or i == 4 or i == 6 or i == 7 or i == 9 or i == 11):
    #                         avg_val_loss_gridded_m[g][k][l][i] = avg_val_loss_gridded_m[g][k][l][i] / 62
    #                     elif (i == 3 or i == 5 or i == 8 or i == 10):
    #                         avg_val_loss_gridded_m[g][k][l][i] = avg_val_loss_gridded_m[g][k][l][i] / 60
    #                     elif (i == 1):
    #                         avg_val_loss_gridded_m[g][k][l][i] = avg_val_loss_gridded_m[g][k][l][i] / 56
    #
    #     print(avg_val_loss_gridded[g])
    #
    #     if(graph):
    #         if(month):
    #             x_axis = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    #             labels = ["t2m", "u10", "v10", 'z', 't', "tcc"]
    #             fig = plt.figure()
    #             for i in range(6):
    #                 plt.plot(avg_val_loss_gridded[i][0][1], label=labels[i])
    #             fig.suptitle('TP analysis, p_x ~ 0, p_other ~ 1 per month')
    #             plt.legend()
    #             plt.xlabel('months')
    #             plt.ylabel('Average loss')
    #             fig.savefig("/home/ge75tis/Desktop/TP_Dropout_per_month_p_x_0")
    #         fig = plt.figure(g, figsize=(10,10))
    #         sns.heatmap(avg_val_loss_gridded[g], linewidths=.5, cmap="Greens", annot=True, xticklabels=grid, yticklabels=grid, norm=LogNorm(), fmt=".3f")
    #         labels = ["t2m", "u10", "v10", 'z', 't', "tcc"]
    #         plt.title('Avg Validation loss of TP for different dropout rates of {par} and other parameters'.format(par=labels[g]))
    #         plt.xlabel('other parameters dropout rate p')
    #         plt.ylabel('{par} dropout rate p'.format(par=labels[g]))
    #         fig.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/DROPOUT_ANALYSIS/tp_analysis_{label}_heatmap'.format(label=labels[g]))


    if(all_individual_models):
        for i, (vals) in enumerate(val_loader):
            for j in range(7):
                valid[j] = vals.index_select(dim=1, index=input_indices[j]).to(device, dtype=torch.float32)
                target[j] = vals.index_select(dim=1, index=out_index[j]).to(device, dtype=torch.float32)

            for j in range(7):
                output[j] = model[j](valid[j])

            for j in range(7):
                loss[j] = loss_type(output[j], target[j])

            for j in range(7):
                val_loss[j] += loss[j].item()

            counter += 1

        for j in range(7):
            avg_val_loss[j] = val_loss[j]/len(val_dataset)
        print(f'Average Validation Losses: {avg_val_loss}')




def test():
    test_loss1 = 0
    for i in range(7):
        model[i].eval()
    for i, (tests) in enumerate(test_loader):
        test = tests.index_select(dim=1, index=input_indices[0]).to(device, dtype=torch.float32)
        target = tests.index_select(dim=1, index=out_index[0]).to(device, dtype=torch.float32)

        output = train_model(test)
        loss = loss_type(output, target)
        test_loss1 += loss.item()

    if(all_individual_models):
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
            avg_test_loss[j] = test_loss[j]/len(test_dataset)

        print(f'Average Test Losses: {avg_test_loss}')
        # log_scalar("a_individual_test_loss", avg_test_loss)


def log_scalar(name, value):
    mlflow.log_metric(name, value)


with mlflow.start_run() as run:
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('lr', learning_rate)
    mlflow.log_param('total_epochs', num_epochs)

    val()
    # test()
    # for epoch in range(num_epochs):
    #     train(epoch)
    #     total_epochs += 1




# train_state = {
#     'epoch': num_epochs,
#     'state_dict': train_model.state_dict(),
#     'optimizer': optimizer.state_dict()
# }
#
# torch.save(train_state, '/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/tp_gaussiandropout_35_epochs')



# clm_losses = [0.1061, 0.5411, 0.6394, 0.2035, 0.1897, 0.6936, 0.3540]
# labels = []
# u10_losses = [0.2416, 0.2759]
# v10_losses = [0.2722, 0.3014]

# labels_0 = ['V10', 'U10', 'TCC', 'Z', 'T', 'T2M']
# p_x_0 = [0.596, 0.649, 0.872, 1.013, 1.064, 1.101]
# # labels_1 = ['T', 'Z', 'V10', 'U10', 'TCC', 'TP']
# p_x_1 = [0.284, 0.288, 0.277, 0.270, 0.269, 0.267]
#
# x = np.arange(len(labels_0))
# width = 0.3
#
# fig3, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, p_x_0, width, label='p_x ~ 0, p_other ~ 1')
# rects2 = ax.bar(x + width/2, p_x_1, width, label='p_x ~ 1, p_other ~ 0')
#
# ax.set_ylabel('Avg validation losses')
# ax.set_title('TP Dropout Analysis noise comparison')
# ax.set_xticks(x, labels_0)
#
# ax.legend()
#
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
#
# fig3.tight_layout()
#
# fig3.savefig('/home/ge75tis/Desktop/TP_Dropout_Analysis')




