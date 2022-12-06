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

train_saved = True

train_model = UNet(6,1).to(device)
train_model.apply(init_weights)
loss_type = nn.L1Loss()
optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

if(train_saved):
    train_state = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/z_gaussiandropout_35_epochs')
    train_model.load_state_dict(train_state['state_dict'])
    optimizer.load_state_dict(train_state['optimizer'])
    total_epochs = train_state['epoch']

model = numpy.empty(7, dtype=UNet)
state = numpy.empty(7, dtype=dict)

# for i in range(7):
#     model[i] = UNet(6,1).to(device)
#
# for i,j in zip(range(7), params):
#     state[i] = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/{parameter}_gaussiandropout_35_epochs'.format(parameter=j))
#
# for i in range(7):
#     model[i].load_state_dict(state[i]['state_dict'])



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
avg_val_loss_gridded = [[[0 for i in range(5)] for j in range(5)] for k in range(6)]

grid = [0.001, 0.25, 0.5, 0.75, 0.999]


# fig1 = plt.figure(figsize=(10, 10))
# avg_val_loss_gridded[0] = [[0.07223247633609053, 0.07964747918591107, 0.0883118561175588, 0.10177320688962936, 0.29970563552150986], [0.07230526194078465, 0.07981637663220706, 0.08823315289432872, 0.10181890119836755, 0.29834342106972656], [0.07239443869011043, 0.08019968682568367, 0.08834550634843029, 0.10210886564973283, 0.29822644948551097], [0.07270217354166998, 0.08029751216814127, 0.08906432890320477, 0.10238040886717299, 0.30084598597190154], [0.09949392193596657, 0.10280341790759401, 0.10788401136659596, 0.11829791715087956, 0.3576698717801538]]
# avg_val_loss_gridded[1] = [[0.07224875334905435, 0.0779249332028709, 0.08340419935547326, 0.09070877009262777, 0.2273634640525465], [0.07417768971764878, 0.07987793488045261, 0.08560993374924954, 0.0929993244576944, 0.22819257378578187], [0.07704354369987364, 0.08290942423555948, 0.08866584375500679, 0.09690072436447013, 0.22989757116118523], [0.08161948560035392, 0.08706140199752703, 0.09341033651200059, 0.10203973511106347, 0.23400969601249041], [0.1297990157383762, 0.13296104807355633, 0.1386829237823617, 0.14689168021695254, 0.3565594781955628]]
# avg_val_loss_gridded[2] = [[0.07224636866315587, 0.07930183222849076, 0.08606606067655838, 0.0952675643951109, 0.2724332095417258], [0.07290584846616607, 0.08003455435168254, 0.08693083384878014, 0.09625155959431439, 0.2729885955583559], [0.07406917438931661, 0.08126747112670174, 0.0885276457932714, 0.09838685466818614, 0.27295801631391864], [0.0761866184946609, 0.08356688084871802, 0.09160931412487813, 0.10255517543178715, 0.2722176436692068], [0.1143069079885744, 0.11856254924037685, 0.12423408647922621, 0.13314516789292635, 0.35642232253943407]]
# avg_val_loss_gridded[3] = [[0.07222851854685235, 0.0751324599277075, 0.0801298595607689, 0.08983190137229553, 0.22767530106110115], [0.0769121135750862, 0.07995449990442355, 0.08494784678396297, 0.09500981578475809, 0.2311676678388086], [0.08043381250782372, 0.083444037349665, 0.08864000877493049, 0.09906585233872883, 0.234540736695675], [0.08357149440131775, 0.08676984301985127, 0.09190361781479561, 0.10236760483008542, 0.24546537046154884], [0.12704726100376207, 0.1286294250965935, 0.1308903208334152, 0.13717921185370993, 0.35541258733974745]]
# avg_val_loss_gridded[4] = [[0.07224432358084476, 0.07962124701844503, 0.08815029963646849, 0.10132545947212063, 0.3387743875180205], [0.07237210507466368, 0.07989414251116041, 0.08843033959808415, 0.10150546774472276, 0.3401382087844692], [0.07259484686671872, 0.0801963963788258, 0.08849894198244565, 0.10220946837982087, 0.3420796982025447], [0.07311084531757929, 0.08066596418925344, 0.08895014480572858, 0.10263245554819499, 0.34016935159898787], [0.09060731204609349, 0.09739976786587336, 0.10367098811962833, 0.11478320765372825, 0.3588205927447097]]
# avg_val_loss_gridded[5] = [[0.07224775938036507, 0.07968858854513462, 0.08806687421994666, 0.10170316623702441, 0.32497815295849763], [0.0723057232695083, 0.07986549262008438, 0.08851047570574773, 0.10214369566473243, 0.32502967760170975], [0.07249487579277117, 0.0799739069165024, 0.0884199210736033, 0.10236625761201937, 0.32287972559259365], [0.07276429666640008, 0.08044768877531568, 0.08906164017238029, 0.10249396533998724, 0.32578530864764566], [0.087986580700907, 0.09416580579868734, 0.10132609239383919, 0.11379031764521991, 0.352410125773247]]
# sns.heatmap(avg_val_loss_gridded[0], linewidths=.5, cmap="Greens", annot=True, square=True, xticklabels=grid,
#                        yticklabels=grid, norm=LogNorm(), fmt=".3f")
# plt.title('Avg Validation loss for different dropout rates of T2M and other parameters')
# plt.xlabel('other parameters dropout rate p')
# plt.ylabel('T2M dropout rate p')
# fig1.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_t2m_heatmap')

# u10_heat = sns.heatmap(avg_val_loss_gridded[1], linewidths=.5, cmap="Greens", annot=True, square=True, xticklabels=grid,
#                        yticklabels=grid, norm=LogNorm())
# v10_heat = sns.heatmap(avg_val_loss_gridded[2], linewidths=.5, cmap="Greens", annot=True, square=True, xticklabels=grid,
#                        yticklabels=grid, norm=LogNorm())
# t_heat = sns.heatmap(avg_val_loss_gridded[3], linewidths=.5, cmap="Greens", annot=True, square=True, xticklabels=grid,
#                      yticklabels=grid, norm=LogNorm())
# tcc_heat = sns.heatmap(avg_val_loss_gridded[4], linewidths=.5, cmap="Greens", annot=True, square=True, xticklabels=grid,
#                        yticklabels=grid, norm=LogNorm())
# tp_heat = sns.heatmap(avg_val_loss_gridded[5], linewidths=.5, cmap="Greens", annot=True, square=True, xticklabels=grid,
#                       yticklabels=grid, norm=LogNorm())

# t2m_heat.get_figure().savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_t2m_heatmap')
# u10_heat.get_figure().savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_u10_heatmap')
# v10_heat.get_figure().savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_v10_heatmap')
# t_heat.get_figure().savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_t_heatmap')
# tcc_heat.get_figure().savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_tcc_heatmap')
# tp_heat.get_figure().savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_tp_heatmap')


def val():
    counter = 0
    val_loss1 = 0
    # for i in range(7):
    #     model[i].eval()

    grid = [0.001, 0.25, 0.5, 0.75, 0.999]

    for g in range(6):
        for k in range(5):
            for l in range(5):
                for i, (vals) in enumerate(val_loader):
                    valid = vals.index_select(dim=1, index=input_indices[3]).to(device, dtype=torch.float32)
                    target = vals.index_select(dim=1, index=out_index[3]).to(device, dtype=torch.float32)

                    p = np.empty([len(valid), 6])
                    if(g == 0):
                        p[0] = [grid[k], grid[l], grid[l], grid[l], grid[l], grid[l]]
                    if(g == 1):
                        p[0] = [grid[l], grid[k], grid[l], grid[l], grid[l], grid[l]]
                    if(g == 2):
                        p[0] = [grid[l], grid[l], grid[k], grid[l], grid[l], grid[l]]
                    if(g == 3):
                        p[0] = [grid[l], grid[l], grid[l], grid[k], grid[l], grid[l]]
                    if(g == 4):
                        p[0] = [grid[l], grid[l], grid[l], grid[l], grid[k], grid[l]]
                    if(g == 5):
                        p[0] = [grid[l], grid[l], grid[l], grid[l], grid[l], grid[k]]
                    tens_p = torch.tensor(p).to(device, dtype=torch.float32)
                    noisy_input = gaussian_dropout(valid, tens_p)

                    output = train_model(noisy_input)
                    loss = loss_type(output, target)
                    val_loss1 += loss.item()

                print(f'Average Validation Losses: {val_loss1/len(val_dataset):.6f}')
                avg_val_loss_gridded[g][k][l] = val_loss1/len(val_dataset)
                val_loss1 = 0

        print(avg_val_loss_gridded[g])

    # plt.figure(figsize=(10,10))
    # t2m_heat = sns.heatmap(avg_val_loss_gridded[0], linewidths=.5, cmap="Greens", annot=True, square=True, xticklabels=grid, yticklabels=grid, norm=LogNorm(), fmt=".3f")
    # u10_heat = sns.heatmap(avg_val_loss_gridded[1], linewidths=.5, cmap="Greens", annot=True, square=True, xticklabels=grid, yticklabels=grid, norm=LogNorm(), fmt=".3f")
    # v10_heat = sns.heatmap(avg_val_loss_gridded[2], linewidths=.5, cmap="Greens", annot=True, square=True, xticklabels=grid, yticklabels=grid, norm=LogNorm(), fmt=".3f")
    # t_heat = sns.heatmap(avg_val_loss_gridded[3], linewidths=.5, cmap="Greens", annot=True, square=True, xticklabels=grid, yticklabels=grid, norm=LogNorm(), fmt=".3f")
    # tcc_heat = sns.heatmap(avg_val_loss_gridded[4], linewidths=.5, cmap="Greens", annot=True, square=True, xticklabels=grid, yticklabels=grid, norm=LogNorm(), fmt=".3f")
    # tp_heat = sns.heatmap(avg_val_loss_gridded[5], linewidths=.5, cmap="Greens", annot=True, square=True, xticklabels=grid, yticklabels=grid, norm=LogNorm(), fmt=".3f")
    #
    # t2m_heat.get_figure().savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_t2m_heatmap')
    # u10_heat.get_figure().savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_u10_heatmap')
    # v10_heat.get_figure().savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_v10_heatmap')
    # t_heat.get_figure().savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_t_heatmap')
    # tcc_heat.get_figure().savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_tcc_heatmap')
    # tp_heat.get_figure().savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_tp_heatmap')

    #     for j in range(7):
            #         valid[j] = vals.index_select(dim=1, index=input_indices[j]).to(device, dtype=torch.float32)
            #         target[j] = vals.index_select(dim=1, index=out_index[j]).to(device, dtype=torch.float32)
            #
            #     for j in range(7):
            #         output[j] = model[j](valid[j])
            #
            #     for j in range(7):
            #         loss[j] = loss_type(output[j], target[j])
            #
            #     for j in range(7):
            #         val_loss[j] += loss[j].item()
            #
            #     counter += 1
            #
            # for j in range(7):
            #     avg_val_loss[j] = val_loss[j]/len(val_dataset)
            # print(f'Average Validation Losses: {avg_val_loss}')


    # log_scalar()

def test():
    test_loss1 = 0
    for i in range(7):
        model[i].eval()
    for i, (tests) in enumerate(test_loader):
        # test = tests.index_select(dim=1, index=input_indices[0]).to(device, dtype=torch.float32)
        # target = tests.index_select(dim=1, index=out_index[0]).to(device, dtype=torch.float32)
        #
        # output = train_model(test)
        # loss = loss_type(output, target)
        # test_loss1 += loss.item()

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

    # val()
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




# labels = ['T2M', 'U10', 'V10', 'Z', 'T', 'TCC', 'TP']
# many_to_one_losses = [0.0618, 0.2434, 0.2759, 0.0543, 0.0822, 0.4916, 0.2332]
# 5 to 1 model U10 Prediction without wind input is 0.2722, +12.6 % difference
# 5 to 1 model V10 Prediction without wind input is 0.3014, +10.7 % difference
# clm_losses = [0.1061, 0.5411, 0.6394, 0.2035, 0.1897, 0.6936, 0.3540]

# labels = ['U10', 'V10']
# u10_losses = [0.2416, 0.2759]
# v10_losses = [0.2722, 0.3014]
#
#
# x = np.arange(len(labels))
# width = 0.3
#
# fig3, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, u10_losses, width, label='other component in input')
# rects2 = ax.bar(x + width/2, v10_losses, width, label='no wind components in input')
#
# ax.set_ylabel('L1 Losses')
# ax.set_title('Comparison of wind component losses')
# ax.set_xticks(x, labels)
#
# ax.legend()
#
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
#
# fig3.tight_layout()
#
# plt.show()
# fig3.savefig('/home/ge75tis/Desktop/test')




