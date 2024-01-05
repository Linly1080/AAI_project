import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from torchvision import transforms
import torch.nn.functional as F

# 设定设备

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        for label in os.listdir(data_dir):
            img_folder = os.path.join(data_dir, label)
            for img_file in os.listdir(img_folder):
                self.samples.append((os.path.join(img_folder, img_file), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.load(img_path)
        image = torch.from_numpy(image).float()

        # if self.transform:
        #     image = self.transform(image)

        return image, label

# 载入数据集
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # 这里可以添加更多的转换操作
# ])

train_dataset = CustomDataset(data_dir='/data/linhuiyan/BIBM2023/AAI_project/new_data/train')
val_dataset = CustomDataset(data_dir='/data/linhuiyan/BIBM2023/AAI_project/new_data/val')

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

# 定义你的模型
class CNN(nn.Module):
    def __init__(self, include_fc, hidden_dim, input_channels=10):
        super(CNN, self).__init__()

        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(9216, hidden_dim)

        self.include_fc = include_fc
        if self.include_fc:
            self.out_dim = hidden_dim
        else:
            self.out_dim = 9216

    def forward(self, input):
        x = input.view(input.shape[0], self.input_channels, 28, 28)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        if self.include_fc:
            x = self.fc1(x)
            x = F.relu(x)

        return x

model = CNN(include_fc=True,hidden_dim=300).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')


train_model(model, criterion, optimizer)
