from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def load_local_lfw_data(data_dir, image_size=(50, 50)):
    images = []
    labels = []
    label_names = sorted(os.listdir(data_dir))
    label_to_id = {name: i for i, name in enumerate(label_names)}

    for label_name in label_names:
        label_dir = os.path.join(data_dir, label_name)
        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, image_size)
                images.append(img_resized)
                labels.append(label_to_id[label_name])

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, label_names

def visualize_img(images, labels, label_names):
    plt.figure(figsize=(10,10))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i], cmap = "gray")
        plt.title(label_names[labels[i]])
        plt.axis("off")
    
    plt.show()

def preprocessing(images, labels):
    # Normalize the pixel values
    images_normalized = images / 255.0

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        images_normalized, labels, test_size=0.2, random_state=42
    )

    print("Training data shape: ", X_train.shape)
    print("Testing data shape: ", X_test.shape)

    return X_train, X_test, y_train, y_test

def feature_extraction(X_train, X_test):
    # Flatten images for PCA
    n_samples, h, w = X_train.shape
    X_train_flattened = X_train.reshape(n_samples, h * w)

    # Apply PCA
    pca = PCA(n_components=150, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flattened)

    # Transform test data
    X_test_flattened = X_test.reshape(X_test.shape[0], h * w)
    X_test_pca = pca.transform(X_test_flattened)

    print("PCA transformed training data shape: ", X_train_pca.shape)

    return X_train_pca, X_test_pca

def classifier(X_train_pca, y_train, X_test_pca, y_test, label_names):
    # Train the classifier
    svm =SVC(kernel="linear", C=1, random_state=42)
    svm.fit(X_train_pca, y_train)

    # Predict on test data
    y_pred = svm.predict(X_test_pca)

    # Get unique classes from y_test
    unique_classes = np.unique(y_test)
    filtered_label_names = [label_names[i] for i in unique_classes]

    # Evaluate the model
    print(classification_report(y_test, y_pred, target_names=filtered_label_names))

    return y_pred

def visualize_pred(X_test, label_names, y_pred, y_test):
    # Plot test images with preditions
    plt.figure(figsize=(10, 10))
    
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X_test[i], cmap="gray")
        pred_name = label_names[y_pred[i]]
        true_name = label_names[y_test[i]]
        color = "green" if y_pred[i] == y_test[i] else "red"
        plt.title(f"Prediction: {pred_name}\nTrue: {true_name}", color=color)
        plt.axis("off")
    
    plt.show()

def main():
    # Load the dataset from local directory
    images, labels, label_names = load_local_lfw_data("lfw-deepfunneled")
    print("Loaded", len(images), "images with", len(label_names), "classes.")

    # Visualize them
    visualize_img(images, labels, label_names)

    # Preprocess the image
    X_train, X_test, y_train, y_test = preprocessing(images, labels)

    # Using PCA to extract the features
    X_train_pca, X_test_pca = feature_extraction(X_train, X_test)

    # Using Support Vector Machine to do the prediction
    y_pred = classifier(X_train_pca, y_train, X_test_pca, y_test, label_names)

    # Visaulize the result
    visualize_pred(X_test, label_names, y_pred, y_test)

if __name__ == '__main__':
    main()