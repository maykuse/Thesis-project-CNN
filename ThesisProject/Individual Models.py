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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 20
batch_size = 10
learning_rate = 0.001
total_epochs = 0

train_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_train')
validation_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_val')
test_set = zarr.open('/home/ge75tis/Desktop/oezyurt/zarr dataset/resampled/resampled_24_test')
# order of the parameters are: t2m, u10, v10, z, t, tcc, tp, lsm, orog, slt

avg_t2m = 0
avg_u10 = 0
avg_v10 = 0
avg_z = 0
avg_t = 0
avg_tcc = 0
avg_tp = 0

indices_t2m = torch.tensor([1,2,3,4,5,6])
t2m_out = torch.tensor([0])
indices_u10 = torch.tensor([0,2,3,4,5,6])
u10_out = torch.tensor([1])
indices_v10 = torch.tensor([0,1,3,4,5,6])
v10_out = torch.tensor([2])
indices_z = torch.tensor([0,1,2,4,5,6])
z_out = torch.tensor([3])
indices_t = torch.tensor([0,1,2,3,5,6])
t_out = torch.tensor([4])
indices_tcc = torch.tensor([0,1,2,3,4,6])
tcc_out = torch.tensor([5])
indices_tp = torch.tensor([0,1,2,3,4,5])
tp_out = torch.tensor([6])

indices_no_wind = torch.tensor([0,3,4,5,6])

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

model = UNet(5,1).to(device)
# model1 = UNet(6,1).to(device)
# model2 = UNet(6,1).to(device)
# model3 = UNet(6,1).to(device)
# model4 = UNet(6,1).to(device)
# model5 = UNet(6,1).to(device)
# model6 = UNet(6,1).to(device)
model.apply(init_weights)

loss_type = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.25)

# state = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/t2m_20_epochs')

# state1 = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/u10_20_epochs')
# state2 = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/v10_20_epochs')
# state3 = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/z_20_epochs')
# state4 = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/t_20_epochs')
# state5 = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/tcc_20_epochs')
# state6 = torch.load('/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/tp_20_epochs')

# model.load_state_dict(state['state_dict'])
# optimizer.load_state_dict(state['optimizer'])
# total_epochs = state['epoch']

# model1.load_state_dict(state1['state_dict'])
# model2.load_state_dict(state2['state_dict'])
# model3.load_state_dict(state3['state_dict'])
# model4.load_state_dict(state4['state_dict'])
# model5.load_state_dict(state5['state_dict'])
# model6.load_state_dict(state6['state_dict'])



# The model architecture is always 6 inputs predicting the 1 remaining output
# changing the indices will change the input/output combinations of the model

y_loss = {}
y_loss['train'] = []
y_loss['val'] = []
x_epoch = []

fig = plt.figure(figsize=(36, 12))
ax0 = fig.add_subplot(121, title="loss")

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'ro-', label='train', markevery=5)
    ax0.plot(x_epoch, y_loss['val'], 'bo-', label='val', markevery=5)
    if current_epoch == 0:
        ax0.legend()
    fig.savefig('/home/ge75tis/Desktop/t2m_loss_curve.jpg', dpi=100)


def train(epoch):
    model.train()
    train_loss = 0
    for i, (inputs) in enumerate(train_loader):
        input = inputs.index_select(dim=1, index=indices_no_wind).to(device, dtype=torch.float32)
        target = inputs.index_select(dim=1, index=v10_out).to(device, dtype=torch.float32)

        optimizer.zero_grad()
        output = model(input)
        loss = loss_type(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(inputs)

    if(epoch >= 5 and epoch % 2 == 0):
        scheduler.step()

    avg_train_loss = train_loss/len(train_dataset)
    print(f'Epoch: {total_epochs}, Average Training Loss: {avg_train_loss:.6f}')
    log_scalar("v10_nowind_individual_train_loss", avg_train_loss)
    y_loss['train'].append(avg_train_loss)



t2m_week_loss = [0 for j in range(52)]
u10_week_loss = [0 for j in range(52)]
v10_week_loss = [0 for j in range(52)]
z_week_loss = [0 for j in range(52)]
t_week_loss = [0 for j in range(52)]
tcc_week_loss = [0 for j in range(52)]
tp_week_loss = [0 for j in range(52)]

clm_month_t2m = [0 for k in range(12)]
clm_month_u10 = [0 for k in range(12)]
clm_month_v10 = [0 for k in range(12)]
clm_month_z = [0 for k in range(12)]
clm_month_t = [0 for k in range(12)]
clm_month_tcc = [0 for k in range(12)]
clm_month_tp = [0 for k in range(12)]

def val():
    index_month = 0
    indx = 0
    ipp = 0
    val_loss = 0
    model.eval()
    for i, (vals) in enumerate(val_loader):
        val = vals.index_select(dim=1, index=indices_no_wind).to(device, dtype=torch.float32)
        target = vals.index_select(dim=1, index=v10_out).to(device, dtype=torch.float32)
        # val1 = vals.index_select(dim=1, index=indices_u10).to(device, dtype=torch.float32)
        # target1 = vals.index_select(dim=1, index=u10_out).to(device, dtype=torch.float32)
        # val2 = vals.index_select(dim=1, index=indices_v10).to(device, dtype=torch.float32)
        # target2 = vals.index_select(dim=1, index=v10_out).to(device, dtype=torch.float32)
        # val3 = vals.index_select(dim=1, index=indices_z).to(device, dtype=torch.float32)
        # target3 = vals.index_select(dim=1, index=z_out).to(device, dtype=torch.float32)
        # val4 = vals.index_select(dim=1, index=indices_t).to(device, dtype=torch.float32)
        # target4 = vals.index_select(dim=1, index=t_out).to(device, dtype=torch.float32)
        # val5 = vals.index_select(dim=1, index=indices_tcc).to(device, dtype=torch.float32)
        # target5 = vals.index_select(dim=1, index=tcc_out).to(device, dtype=torch.float32)
        # val6 = vals.index_select(dim=1, index=indices_tp).to(device, dtype=torch.float32)
        # target6 = vals.index_select(dim=1, index=tp_out).to(device, dtype=torch.float32)

        output = model(val)
        # output1 = model1(val1)
        # output2 = model2(val2)
        # output3 = model3(val3)
        # output4 = model4(val4)
        # output5 = model5(val5)
        # output6 = model6(val6)

        loss = loss_type(output, target)
        val_loss += loss.item()
        # loss1 = loss_type(output1, target1)
        # loss2 = loss_type(output2, target2)
        # loss3 = loss_type(output3, target3)
        # loss4 = loss_type(output4, target4)
        # loss5 = loss_type(output5, target5)
        # loss6 = loss_type(output6, target6)

        ipp += 1

    avg_val_loss = val_loss/len(val_dataset)
    print(f'Average Validation Loss: {avg_val_loss:.6f}')
    log_scalar("v10_nowind_individual_val_loss", avg_val_loss)
    y_loss['val'].append(avg_val_loss)
    # draw_curve(total_epochs)


def test():
    test_loss = 0
    model.eval()
    for i, (tests) in enumerate(test_loader):
        test = tests.index_select(dim=1, index=indices_no_wind).to(device, dtype=torch.float32)
        target = tests.index_select(dim=1, index=v10_out).to(device, dtype=torch.float32)

        output = model(test)
        loss_t = loss_type(output, target)
        test_loss += loss_t.item()

    avg_test_loss = test_loss/len(test_dataset)
    print(f'Average Test Loss at the End: {avg_test_loss:.6f}')
    log_scalar("v10_nowind_individual_test_loss", avg_test_loss)


def log_scalar(name, value):
    mlflow.log_metric(name, value)


with mlflow.start_run() as run:
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('lr', learning_rate)
    mlflow.log_param('total_epochs', num_epochs)

    # for epoch in range(num_epochs):
    #     train(epoch)
    #     val()
    #     total_epochs += 1
    #
    # test()




# state = {
#     'epoch': num_epochs,
#     'state_dict': model.state_dict(),
#     'optimizer': optimizer.state_dict()
# }
#
# torch.save(state, '/home/ge75tis/Desktop/oezyurt/model/6_1_UNET/v10_nowind_20_epochs')


# labels = ['T2M', 'U10', 'V10', 'Z', 'T', 'TCC', 'TP']
# many_to_one_losses = [0.0618, 0.2416, 0.2759, 0.0543, 0.0822, 0.4916, 0.2332]
# 5 to 1 model U10 Prediction without wind input is 0.2722, +12.6 % difference
# 5 to 1 model V10 Prediction without wind input is 0.3014, +10.7 % difference
# clm_losses = [0.1061, 0.5411, 0.6394, 0.2035, 0.1897, 0.6936, 0.3540]

# labels = ['U10', 'V10']
# u10_losses = [0.2416, 0.2722]
# v10_losses = [0.2759, 0.3014]


# x = np.arange(len(labels))
# width = 0.3
#
# fig3, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, many_to_one_losses, width, label='many-to-one model')
# rects2 = ax.bar(x + width/2, clm_losses, width, label='climatology model')
#
# ax.set_ylabel('L1 Losses per param')
# ax.set_title('Comparison of climatology and many-to-one model predictions')
# ax.set_xticks(x, labels)
# ax.legend()
#
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
#
# fig3.tight_layout()
#
# plt.show()
# fig3.savefig('/home/ge75tis/Desktop/test')




# for i in range(52):
#      if(i == 51):
#         t2m_week_loss[i] = t2m_week_loss[i] / 16
#         u10_week_loss[i] = u10_week_loss[i] / 16
#         v10_week_loss[i] = v10_week_loss[i] / 16
#         z_week_loss[i] = z_week_loss[i] / 16
#         t_week_loss[i] = t_week_loss[i] / 16
#         tcc_week_loss[i] = tcc_week_loss[i] / 16
#         tp_week_loss[i] = tp_week_loss[i] / 16
#      else:
#         t2m_week_loss[i] = t2m_week_loss[i] / 14
#         u10_week_loss[i] = u10_week_loss[i] / 14
#         v10_week_loss[i] = v10_week_loss[i] / 14
#         z_week_loss[i] = z_week_loss[i] / 14
#         t_week_loss[i] = t_week_loss[i] / 14
#         tcc_week_loss[i] = tcc_week_loss[i] / 14
#         tp_week_loss[i] = tp_week_loss[i] / 14
#
# for i in range(12):
#     if(i == 0 or i == 2 or i == 4 or i == 6 or i == 7 or i == 9 or i == 11):
#         clm_month_t2m[i] = clm_month_t2m[i] / 62
#         clm_month_u10[i] = clm_month_u10[i] / 62
#         clm_month_v10[i] = clm_month_v10[i] / 62
#         clm_month_z[i] = clm_month_z[i] / 62
#         clm_month_t[i] = clm_month_t[i] / 62
#         clm_month_tcc[i] = clm_month_tcc[i] / 62
#         clm_month_tp[i] = clm_month_tp[i] / 62
#     elif(i == 3 or i == 5 or i == 8 or i == 10):
#         clm_month_t2m[i] = clm_month_t2m[i] / 60
#         clm_month_u10[i] = clm_month_u10[i] / 60
#         clm_month_v10[i] = clm_month_v10[i] / 60
#         clm_month_z[i] = clm_month_z[i] / 60
#         clm_month_t[i] = clm_month_t[i] / 60
#         clm_month_tcc[i] = clm_month_tcc[i] / 60
#         clm_month_tp[i] = clm_month_tp[i] / 60
#     elif(i == 1):
#         clm_month_t2m[i] = clm_month_t2m[i] / 56
#         clm_month_u10[i] = clm_month_u10[i] / 56
#         clm_month_v10[i] = clm_month_v10[i] / 56
#         clm_month_z[i] = clm_month_z[i] / 56
#         clm_month_t[i] = clm_month_t[i] / 56
#         clm_month_tcc[i] = clm_month_tcc[i] / 56
#         clm_month_tp[i] = clm_month_tp[i] / 56
#
# fig = plt.figure()
# plt.plot(clm_month_t2m, label="t2m")
# plt.plot(clm_month_u10, label="u10")
# plt.plot(clm_month_v10, label="v10")
# plt.plot(clm_month_z, label="z")
# plt.plot(clm_month_t, label="t")
# plt.plot(clm_month_tcc, label="tcc")
# plt.plot(clm_month_tp, label="tp")
# plt.legend(loc='center right')
#
# fig.suptitle('many-to-one model prediction L1 loss by month per parameter')
# plt.xlabel('calendar months')
# plt.ylabel('Average loss')
# plt.show()
# fig.savefig("/home/ge75tis/Desktop/per_param_unet_loss_by_month_L1")