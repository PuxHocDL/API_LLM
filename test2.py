import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
import cv2

# Danh sách tên các lớp
class_names = [
    'Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast', 'Leaf Scald',
    'Narrow Brown Leaf Spot', 'Neck Blast', 'Rice Hispa', 'Sheath Blight', 'Tungro'
]

# Khởi tạo mô hình ResNet-152
model = models.resnet152(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 10 lớp đầu ra

# Load checkpoint
checkpoint_dir = "resnet152.pt"
checkpoint = torch.load(checkpoint_dir, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Hàm để xử lý ảnh và dự đoán
def predict_image(image_path, model, class_names):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()], image_tensor

# Hàm để tính Grad-CAM
def grad_cam(model, image_tensor, target_layer):
    # Lấy đặc trưng từ lớp cuối cùng
    features = model.conv1(image_tensor)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)
    features = model.layer1(features)
    features = model.layer2(features)
    features = model.layer3(features)
    features = model.layer4(features)

    # Giữ gradient của features
    features.retain_grad()

    # Tính gradient của đầu ra đối với các đặc trưng
    outputs = model.avgpool(features)
    outputs = torch.flatten(outputs, 1)
    outputs = model.fc(outputs)
    _, predicted = torch.max(outputs, 1)
    outputs[:, predicted].backward()

    # Lấy gradient và tính trọng số
    gradients = features.grad
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Tính weighted sum của các đặc trưng
    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_gradients[i]

    # Tính heatmap
    heatmap = torch.mean(features, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().numpy(), 0)  # Chuyển đổi sang NumPy
    heatmap /= np.max(heatmap)  # Sử dụng np.max() thay vì torch.max()

    return heatmap
# Đường dẫn đến thư mục chứa dataset
dataset_dir = "dataset_rice"

# Lấy danh sách các ảnh và nhãn
image_paths = []
true_labels = []
for label, class_name in enumerate(class_names):
    class_dir = os.path.join(dataset_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_paths.append(os.path.join(class_dir, image_name))
        true_labels.append(label)

# Giới hạn số lượng ảnh muốn hiển thị
num_images_to_show = 50

# Dự đoán nhãn cho từng ảnh và hiển thị Grad-CAM
for i, image_path in enumerate(image_paths[:num_images_to_show]):
    predicted_class, image_tensor = predict_image(image_path, model, class_names)
    heatmap = grad_cam(model, image_tensor, model.layer4)

    # Đọc ảnh gốc
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    # Resize heatmap để phù hợp với kích thước ảnh
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Kết hợp ảnh gốc và heatmap
    superimposed_img = heatmap * 0.4 + image
    superimposed_img = np.uint8(superimposed_img)

    # Hiển thị ảnh gốc và ảnh với heatmap
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Original Image\nTrue: {class_names[true_labels[i]]}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')

    plt.show()