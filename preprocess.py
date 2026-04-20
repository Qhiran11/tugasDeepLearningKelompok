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
                img = img / 255.0

                data.append(img)
                labels.append(label)
            except:
                continue

    return np.array(data), np.array(labels)


if __name__ == "__main__":
    X_train, y_train = load_dataset("dataset/train")
    X_test, y_test = load_dataset("dataset/test")

    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)
