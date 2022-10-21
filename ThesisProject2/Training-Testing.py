from UNET import *
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader


import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
batch_size = 10
learning_rate = 0.005

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_path = "/home/ge75tis/Downloads/AirQualityUCI/AirQualityUCI.csv"
# We need a 2D-multivariate-timeseries data as input for this UNET, this is a 1D-multivariate-timeseries data


class TestDataset(Dataset):
    def __init__(self):
        air_numpy = np.loadtxt(data_path, dtype=np.int32, delimiter=";", skiprows=1, usecols=(3, 4, 6, 7, 8, 9, 10, 11))
        self.air_torch = torch.from_numpy(air_numpy)

    def __getitem__(self, index):
        return self.air_torch[index]

    def __len__(self):
        return self.air_torch.shape[0]


train_dataset = TestDataset()
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

#test_loader = 0

model = UNet(8, 8).to(device)

loss_type = nn.L1Loss()
# optimizer algorithm may be changed to see different results
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)


for epoch in range(num_epochs):
    for i, (inputs) in enumerate(train_loader):
        inputs = inputs.to(device)

        outputs = model(inputs)
        loss = loss_type(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Training finished!')

