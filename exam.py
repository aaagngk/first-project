import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import logging

# 加载预训练的ResNet50模型
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)  # 输出4类
# 配置日志记录器
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 包装模型
class ResNetWrapper(nn.Module):
    def __init__(self, model):
        super(ResNetWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

model = ResNetWrapper(model)

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root='D:\LDL\data\Classification', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 初始化损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型移动到设备上
model.train()  # 设置为训练模式
for epoch in range(10):  # 举例训练10轮
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels) # 使用输出的键值“out”
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000个batch打印一次损失
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            avg_loss = running_loss / 2000
            logging.info('Epoch: %d, Batch: %d, Loss: %.3f' % (epoch + 1, i + 1, avg_loss))
            running_loss = 0.0

# 设置为评估模式
model.eval()

print('Finished Training')
torch.save(model.state_dict(), 'D:\\LDL\\resnet50.pth')