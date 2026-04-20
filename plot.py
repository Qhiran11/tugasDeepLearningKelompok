import matplotlib.pyplot as plt
import numpy as np

models = ["1C-ReLU", "2C-ReLU", "3C-ReLU", "1C-SIG", "2C-SIG", "3C-SIG"]

training_time = [5.90, 8.30, 9.68, 6.92, 10.25, 11.55]

train_acc = [82.05, 69.46, 51.17, 51.17, 51.68, 48.83]
test_acc = [84.07, 71.68, 50.88, 50.88, 50.88, 49.12]

plt.figure()
plt.bar(models, training_time)
plt.title("Perbandingan Waktu Training (10 Epochs)")
plt.xlabel("Model")
plt.ylabel("Waktu (detik)")
plt.xticks(rotation=30)

for i, v in enumerate(training_time):
    plt.text(i, v + 0.3, str(v), ha="center")

plt.tight_layout()
plt.show()

x = np.arange(len(models))
width = 0.35

plt.figure()
plt.bar(x - width / 2, train_acc, width, label="Train Acc")
plt.bar(x + width / 2, test_acc, width, label="Test Acc")

plt.title("Perbandingan Akurasi (Train vs Test)")
plt.xlabel("Model")
plt.ylabel("Akurasi (%)")
plt.xticks(x, models, rotation=30)
plt.legend()

for i, v in enumerate(train_acc):
    plt.text(i - width / 2, v + 1, str(v), ha="center")

for i, v in enumerate(test_acc):
    plt.text(i + width / 2, v + 1, str(v), ha="center")

plt.tight_layout()
plt.show()
