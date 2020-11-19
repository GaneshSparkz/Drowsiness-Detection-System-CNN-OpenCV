import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer


lb = LabelBinarizer()

# Preprocess train images
print("[INFO] Preprocessing train images...")
train_dir = "dataset/train"
train_images = []
train_labels = []

for label in os.listdir(train_dir):
    img_dir = train_dir + '/' + label
    for filename in os.listdir(img_dir):
        image = cv2.imread(img_dir + '/' + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (24, 24))
        image = image.astype('float') / 255.0
        train_images.append(image)
        train_labels.append(label)

X_train = np.array(train_images)
y_train = np.array(train_labels)

X_train = X_train.reshape(X_train.shape[0], 24, 24, 1)
y_train = lb.fit_transform(y_train)

np.save('data/X_train.npy', X_train)
np.save('data/y_train.npy', y_train)

print(len(train_images), "train images...")


# Preprocess test images
print("[INFO] Preprocessing test images...")
test_dir = "dataset/test"
test_images = []
test_labels = []

for label in os.listdir(test_dir):
    img_dir = test_dir + '/' + label
    for filename in os.listdir(img_dir):
        image = cv2.imread(img_dir + '/' + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (24, 24))
        image = image.astype('float') / 255.0
        test_images.append(image)
        test_labels.append(label)

X_test = np.array(test_images)
y_test = np.array(test_labels)

X_test = X_test.reshape(X_test.shape[0], 24, 24, 1)
y_test = lb.transform(y_test)

np.save('data/X_test.npy', X_test)
np.save('data/y_test.npy', y_test)

print(len(test_images), "test images...")
print("Classes:", lb.classes_)
