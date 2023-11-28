from torchvision import transforms, models
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# 加载模型
class_labels = ['class1', 'class2', 'class3', 'class4']

# 创建并加载预训练的ResNet-50模型
model = models.resnet50()

# 加载保存的模型参数
model.load_state_dict(torch.load("D:/LDL/resnet50.pth"))

# 设置模型为评估模式
model.eval()

# 使用模型进行预测，数据处理
image1 = Image.open(r'./sample\levle3_22.jpg')

# 定义转换操作，先调整图像大小，然后转换为张量，最后进行归一化
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                         std=[0.2814769, 0.226306, 0.20132513]),
])
# 对图像进行转换操作
transformed_image1 = transform(image1)

img1 = transformed_image1.unsqueeze(0)

outputs = model(img1)
_, predicted = torch.max(outputs, 1)

# 打印预测结果
print("Predicted label:", class_labels[predicted.item()])