from sklearn.datasets import fetch_openml

# Load MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Chọn một số ảnh từ tập dữ liệu
for i in range(5):  # Show 5 ảnh
    plt.figure(figsize=(2,2))
    plt.imshow(X[i].reshape(28,28), cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
    plt.savefig(f'mnist_sample_{i}.png')  # Lưu ảnh
    plt.close()