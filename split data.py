import os
import random
import shutil


def split_data(src, train_dst, test_dst, split_ratio=0.8):
    files = os.listdir(src)
    random.shuffle(files)

    split_index = int(len(files) * split_ratio)
    train_files = files[:split_index]
    test_files = files[split_index:]

    for f in train_files:
        shutil.copy(os.path.join(src, f), os.path.join(train_dst, f))

    for f in test_files:
        shutil.copy(os.path.join(src, f), os.path.join(test_dst, f))


os.makedirs("dataset/train/mobil", exist_ok=True)
os.makedirs("dataset/train/motor", exist_ok=True)
os.makedirs("dataset/test/mobil", exist_ok=True)
os.makedirs("dataset/test/motor", exist_ok=True)

split_data("mobil", "dataset/train/mobil", "dataset/test/mobil")
split_data("motor", "dataset/train/motor", "dataset/test/motor")
