import numpy as np
import time
import os
import matplotlib.pyplot as plt
from core import *


def build_model(num_convs, activation_name):
    layers = []
    in_channels = 1

    out_channels = 2

    current_size = 64

    for i in range(num_convs):

        layers.append(
            Conv2D(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        )

        if activation_name == "relu":
            layers.append(ReLU())
        else:
            layers.append(Sigmoid())

        layers.append(MaxPool2D(pool_size=2, stride=2))
        current_size //= 2

        in_channels = out_channels
        out_channels *= 2

    layers.append(Flatten())
    flatten_size = in_channels * current_size * current_size

    layers.append(Dense(flatten_size, 16))
    layers.append(ReLU())
    layers.append(Dense(16, 1))
    layers.append(Sigmoid())

    return layers


def train_model(layers, X_train, y_train, epochs=5, lr=0.01, batch_size=32):
    N = len(X_train)
    start_time = time.time()

    for epoch in range(epochs):
        permutation = np.random.permutation(N)
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]

        epoch_loss = 0
        batches = 0
        for i in range(0, N, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]

            output = X_batch
            for layer in layers:
                output = layer.forward(output)

            loss = binary_cross_entropy(y_batch, output)
            epoch_loss += loss
            batches += 1

            grad = binary_cross_entropy_prime(y_batch, output)
            for layer in reversed(layers):

                grad = layer.backward(grad, lr)

    training_time = time.time() - start_time
    return training_time


def evaluate_model(layers, X, y):
    output = X
    for layer in layers:
        output = layer.forward(output)

    predictions = (output > 0.5).astype(int)
    accuracy = np.mean(predictions == y)

    tp = np.sum((predictions == 1) & (y == 1))
    tn = np.sum((predictions == 0) & (y == 0))
    fp = np.sum((predictions == 1) & (y == 0))
    fn = np.sum((predictions == 0) & (y == 1))

    return accuracy, (tp, tn, fp, fn)


if __name__ == "__main__":
    from preprocess import load_dataset

    print("[INFO] Mengimpor Data dari dataset/train dan dataset/test ...")
    X_train, y_train = load_dataset("dataset/train")
    X_test, y_test = load_dataset("dataset/test")

    X_train = X_train.reshape(-1, 1, 64, 64)
    X_test = X_test.reshape(-1, 1, 64, 64)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print(f"Data Splitting: Train {len(X_train)} | Test {len(X_test)}")

    configs = [
        (1, "relu"),
        (2, "relu"),
        (3, "relu"),
        (1, "sigmoid"),
        (2, "sigmoid"),
        (3, "sigmoid"),
    ]

    results = []

    with open("ablation_results.txt", "w") as f:
        f.write("Ablation Study Results\n")
        f.write("========================\n\n")

        for num_conv, act in configs:
            print(
                f"[EKSPERIMEN] Model: {num_conv} Conv Layer + {act.upper()} Activation"
            )
            model = build_model(num_convs=num_conv, activation_name=act)

            t_time = train_model(
                model, X_train, y_train, epochs=10, lr=0.01, batch_size=32
            )

            train_acc, _ = evaluate_model(model, X_train, y_train)
            test_acc, conf_mat = evaluate_model(model, X_test, y_test)
            tp, tn, fp, fn = conf_mat

            res_str = (
                f"Model: {num_conv}x Conv + {act.upper()}\n"
                f" - Waktu Training (10 Epochs): {t_time:.2f} detik\n"
                f" - Train Akurasi: {train_acc*100:.2f}%\n"
                f" - Test Akurasi: {test_acc*100:.2f}%\n"
                f" - Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}\n\n"
            )

            print(res_str)
            f.write(res_str)

        results.append(
            {
                "label": f"{num_conv}C\n{act[:3].upper()}",
                "train_time": t_time,
                "train_acc": train_acc * 100,
                "test_acc": test_acc * 100,
            }
        )

print("[SELESAI] Hasil eksperimen komparatif telah disimpan di 'ablation_results.txt'.")

labels = [r["label"] for r in results]
train_times = [r["train_time"] for r in results]
train_accs = [r["train_acc"] for r in results]
test_accs = [r["test_acc"] for r in results]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width / 2, train_accs, width, label="Train Acc", color="skyblue")
rects2 = ax.bar(x + width / 2, test_accs, width, label="Test Acc", color="salmon")

ax.set_ylabel("Accuracy (%)")
ax.set_title("Perbandingan Akurasi (Train vs Test)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.savefig("akurasi_plot.png")
print("Disimpan: akurasi_plot.png")

fig2, ax2 = plt.subplots(figsize=(8, 5))
rects3 = ax2.bar(x, train_times, 0.5, color="orange")
ax2.set_ylabel("Waktu (detik)")
ax2.set_title("Perbandingan Waktu Training (10 Epochs)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
plt.tight_layout()
plt.savefig("waktu_training_plot.png")
print("Disimpan: waktu_training_plot.png")
