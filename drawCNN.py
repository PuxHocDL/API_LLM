import torch
import torchvision.models as models
from torchsummary import summary
from torchviz import make_dot

# Kiểm tra xem GPU có khả dụng không
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load mô hình ResNet-152 đã được pre-trained và chuyển về GPU
model = models.resnet152(pretrained=True).to(device)

# Hiển thị tóm tắt mô hình
summary(model, (3, 224, 224))

# Tạo một input giả và chuyển về GPU
x = torch.randn(1, 3, 224, 224).to(device)

# Vẽ mô hình (nếu sử dụng torchviz)
y = model(x)
dot = make_dot(y, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('resnet152_model')