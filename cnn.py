import os
import cv2
import numpy as np

IMG_SIZE = 64


def load_dataset(path):
    data = []
    labels = []

    for label_name in ["mobil", "motor"]:
        folder = os.path.join(path, label_name)
        label = 0 if label_name == "mobil" else 1

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)

            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.astype(np.float32) / 255.0

                data.append(img)
                labels.append(label)
            except:
                continue

    return np.array(data), np.array(labels)


def conv2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1

    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = image[i : i + kh, j : j + kw]
            output[i, j] = np.sum(region * kernel)

    return output


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def flatten(x):
    return x.flatten()


def fully_connected(x, w, b):
    return np.dot(x, w) + b


def loss_fn(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


X_train, y_train = load_dataset("dataset/train")

print("Data:", X_train.shape)

kernel = np.random.randn(3, 3) * 0.01

fc_input_size = 62 * 62
w = np.random.randn(fc_input_size) * 0.01
b = 0.0

lr = 0.001
epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X_train)):
        x = conv2d(X_train[i], kernel)
        x = relu(x)
        x = flatten(x)

        z = fully_connected(x, w, b)
        y_pred = sigmoid(z)

        y_true = y_train[i]
        loss = loss_fn(y_true, y_pred)
        total_loss += loss

        dz = y_pred - y_true

        dw = (x * dz) / len(x)
        db = dz

        w -= lr * dw
        b -= lr * db

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train)}")

sample_image = X_train[0]

x = conv2d(sample_image, kernel)
x = relu(x)
x = flatten(x)
x = fully_connected(x, w, b)
output = sigmoid(x)

print("Sample Output:", output)
