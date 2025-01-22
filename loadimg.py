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
    return class_names[predicted.item()]

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

# Dự đoán nhãn cho từng ảnh
predicted_labels = []
for image_path in image_paths:
    predicted_class = predict_image(image_path, model, class_names)
    predicted_labels.append(class_names.index(predicted_class))

# Tính toán confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Vẽ confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Lưu confusion matrix dưới dạng ảnh
confusion_matrix_image_path = "confusion_matrix.png"
plt.savefig(confusion_matrix_image_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Confusion matrix đã được lưu tại: {confusion_matrix_image_path}")

# Tính toán báo cáo phân loại
report = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)

# Chuyển đổi báo cáo thành DataFrame
report_df = pd.DataFrame(report).transpose()

# Lưu báo cáo dưới dạng file CSV
report_csv_path = "classification_report.csv"
report_df.to_csv(report_csv_path, index=True)

print(f"Báo cáo phân loại đã được lưu tại: {report_csv_path}")

# Vẽ biểu đồ cho Precision, Recall, và F1-score
plt.figure(figsize=(12, 8))

# Vẽ Precision
plt.subplot(3, 1, 1)
plt.bar(class_names, report_df['precision'][:-3], color='blue')
plt.title('Precision')
plt.xticks(rotation=90)

# Vẽ Recall
plt.subplot(3, 1, 2)
plt.bar(class_names, report_df['recall'][:-3], color='green')
plt.title('Recall')
plt.xticks(rotation=90)

# Vẽ F1-score
plt.subplot(3, 1, 3)
plt.bar(class_names, report_df['f1-score'][:-3], color='orange')
plt.title('F1-score')
plt.xticks(rotation=90)

# Hiển thị biểu đồ
plt.tight_layout()

# Lưu biểu đồ dưới dạng ảnh
metrics_plot_path = "metrics_plot.png"
plt.savefig(metrics_plot_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Biểu đồ các chỉ số đã được lưu tại: {metrics_plot_path}")