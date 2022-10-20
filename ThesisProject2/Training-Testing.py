from UNET import *

import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
batch_size = 10
learning_rate = 0.005

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = 0
test_dataset = 0
train_loader = 0
test_loader = 0

model = UNet().to(device)

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

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)

    #   for i in range(batch_size):
        #    print(inputs[i]-outputs[i])

