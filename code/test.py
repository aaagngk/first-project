from torchvision import transforms
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import cv2
#path = "C:\\Users\\14920\\.cache\\torch\\hub\\checkpoints\\resnet50-19c8e357.pth"

# 加载模型
model = models.resnet50(pretrained=True)
model.eval()  # 将模型设置为评估模式，关闭dropout等影响结果的因素

# 加载保存的模型参数
state_dict = torch.load("C:\\Users\\14920\\.cache\\torch\\hub\\checkpoints\\resnet50-19c8e357.pth")
model.load_state_dict(state_dict)

# 使用模型进行预测，数据处理
image1 = Image.open('D:\\LDL\\levle3_127.jpg')
image2 = Image.open('D:\\LDL\\1.jpg')
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
transformed_image2 = transform(image2)
img2 = transformed_image2.unsqueeze(0)
#输出
output1 = model(img1)
output_tensor1 = output1.detach().numpy()
print(output_tensor1)
output2 = model(img2)
output_tensor2 = output2.detach().numpy()
print(output_tensor2)

image = Image.fromarray(output_tensor2.astype(np.uint8))
image.show()
