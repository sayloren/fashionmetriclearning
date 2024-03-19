import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import torch
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_mutual_info_score


def plot_embeddings_comparison(matched_model, train_loader, val_loader):
    # Define label names for FashionMNIST
    label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def visualize_embeddings(loader, label_names, title, ax):
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                anchor_imgs, positive_imgs, negative_imgs = batch[:3]
                output1, output2, output3 = matched_model(anchor_imgs, positive_imgs, negative_imgs)
                embeddings.extend([output1.numpy(), output2.numpy(), output3.numpy()])
                labels.extend([batch[3].numpy(), batch[4].numpy(), batch[5].numpy()])  # Assuming the anchor, positive, and negative labels are at index 3, 4, and 5

        # Concatenate embeddings and labels
        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)

        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot embeddings on the provided axes
        for label_idx, label_name in enumerate(label_names):
            ax.scatter(embeddings_2d[labels == label_idx, 0], embeddings_2d[labels == label_idx, 1], label=label_name)
        ax.set_title(title)
        ax.legend()
        return embeddings
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Visualize embeddings for training set
    training_embeddings = visualize_embeddings(train_loader, label_names, 'Projected Embeddings for Training Set', axes[0])

    # Visualize embeddings for validation set
    validation_embeddings = visualize_embeddings(val_loader, label_names, 'Projected Embeddings for Validation Set', axes[1])

    # Save the embeddings to a figure
    plt.tight_layout()
    plt.title('Comparison of Train and Validation Set Embeddings')
    save_path = os.path.join('plots','embeddings_comparison.png')
    plt.savefig(save_path)
    plt.close()

    return validation_embeddings

# Define a function to extract and visualize embeddings
def visualize_embeddings_with_images(loader, model, labels_to_display):
    plt.figure(figsize=(8,6))

    # Keep track of the number of embeddings displayed
    num_embeddings_displayed = 0

    with torch.no_grad():
        for batch in loader:
            anchor_imgs, _, _ = batch[:3]  # Only need anchor images
            outputs = model.forward_once(anchor_imgs)
            for embedding, label, image in zip(outputs, batch[3], anchor_imgs):
                # Check if the label matches the ones we want to display
                if label.item() in labels_to_display:
                    # Plot the embedding
                    plt.scatter(embedding[0], embedding[1], color='blue', label=f'Label: {label.item()}')

                    # Convert PyTorch tensor to NumPy array
                    image_np = image.squeeze().cpu().numpy()
                    # Resize image using OpenCV
                    resized_image = cv2.resize(image_np, (32, 32))

                    # Create an AnnotationBbox with the image
                    imagebox = OffsetImage(resized_image, zoom=1)
                    ab = AnnotationBbox(imagebox, (embedding[0], embedding[1]), frameon=False)
                    plt.gca().add_artist(ab)

                    num_embeddings_displayed += 1

            # Break if displayed enough embeddings
            if num_embeddings_displayed == len(labels_to_display):
                break

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # plt.legend()
    plt.title('Projected Embeddings with Images for Training Set (Clusters {0})'.format(labels_to_display))
    save_path = os.path.join('plots','embeddings_images.png')
    plt.savefig(save_path)
    plt.close()
    
def plot_pairwise_distance_histogram(embeddings, bins=50):
    """
    Plot a histogram of pairwise distances between embeddings.

    Parameters:
        embeddings (numpy.ndarray): Array of shape (N, d) containing embeddings,
                                    where N is the number of samples and d is the dimensionality of the embeddings.
        bins (int): Number of bins for the histogram. Default is 50.
    """
    # Compute pairwise distances
    pairwise_distances = np.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            distance = np.linalg.norm(embeddings[i] - embeddings[j])  # Euclidean distance
            pairwise_distances[i, j] = distance
            pairwise_distances[j, i] = distance  # Distance matrix is symmetric

    # Flatten the upper triangular part of the distance matrix (excluding the diagonal)
    flat_distances = pairwise_distances[np.triu_indices(len(embeddings), k=1)]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flat_distances, bins=bins, color='skyblue', edgecolor='black')
    plt.title('Histogram of Pairwise Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True)
    save_path = os.path.join('plots','embeddings_pairwise_distances.png')
    plt.savefig(save_path)
    plt.close()

def plot_pairwise_distance_histogram_by_class(embeddings, bins=50, figsize=(11,9)):
    """
    Plot histograms of pairwise distances between embeddings within each class.

    Parameters:
        embeddings (numpy.ndarray): Array of shape (N, d) containing embeddings,
                                    where N is the number of samples and d is the dimensionality of the embeddings.
        labels (numpy.ndarray): Array of shape (N,) containing class labels for each sample.
        bins (int): Number of bins for the histogram. Default is 50.
    """
    # Define label names for FashionMNIST
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    num_rows = int(np.ceil(num_classes / 3))
    num_cols = min(num_classes, 3)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.ravel()

    for i, class_label in enumerate(unique_classes):
        class_indices = labels.index(class_label) # np.where(labels == class_label)[0]
        class_embeddings = embeddings[class_indices]

        # Compute pairwise distances within the class
        pairwise_distances = []
        for j in range(len(class_embeddings)):
            for k in range(j + 1, len(class_embeddings)):
                distance = np.linalg.norm(class_embeddings[j] - class_embeddings[k])  # Euclidean distance
                pairwise_distances.append(distance)

        # Plot histogram in the corresponding subplot
        axes[i].hist(pairwise_distances, bins=bins, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Class {class_label}')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)

    for j in range(num_classes, num_rows * num_cols):
        fig.delaxes(axes[j])

    fig.suptitle('Pairwise Distance Histograms by Class')
    plt.tight_layout()
    save_path = os.path.join('plots','embeddings_pairwise_distance_by_class.png')
    plt.savefig(save_path)
    plt.close()

def plot_pca_embeddings(embeddings, figsize=(8, 6)):
    """
    Perform PCA on embeddings and plot the data points in the reduced 2D space.

    Parameters:
        embeddings (numpy.ndarray): Array of shape (N, d) containing embeddings,
                                    where N is the number of samples and d is the dimensionality of the embeddings.
        labels (numpy.ndarray): Array of shape (N,) containing class labels for each sample.
        figsize (tuple): Size of the figure. Default is (8, 6).
    """

    # Perform PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot PCA embeddings
    plt.figure(figsize=figsize)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
    plt.title('PCA of Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Class Label')
    plt.grid(True)
    
    save_path = os.path.join('plots','embeddings_pca.png')
    plt.savefig(save_path)
    plt.close()

def plot_class_embeddings_similarity(triplet_dataset, matched_model):
    # Initialize a dictionary to store embeddings grouped by class
    class_embeddings = {i: [] for i in range(10)}  # Assuming there are 10 classes

    # Define label names for FashionMNIST
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Compute embeddings for all triplets in the dataset
    with torch.no_grad():
        for triplet in triplet_dataset:
            anchor_img, positive_img, negative_img, anchor_label, _, _ = triplet
            anchor_emb = matched_model.forward_once(anchor_img.unsqueeze(0))
            # Group the anchor embeddings by class
            class_embeddings[anchor_label].append(anchor_emb.flatten().numpy())

    # Convert the embeddings to numpy arrays
    for label in class_embeddings:
        class_embeddings[label] = np.array(class_embeddings[label])

    # Compute pairwise cosine similarity between embeddings within each class
    class_pairwise_similarity = {}
    for label in class_embeddings:
        embeddings = class_embeddings[label]
        pairwise_similarity = cosine_similarity(embeddings)
        class_pairwise_similarity[label] = pairwise_similarity

    # Visualize the similarity matrix for each class
    plt.figure(figsize=(15, 10))
    for label in class_pairwise_similarity:
        plt.subplot(2, 5, label+1)  # Assuming there are 10 classes
        sns.heatmap(class_pairwise_similarity[label], cmap='viridis', annot=False, fmt=".2f")
        plt.title(f'{labels[label]} - Pairwise Cosine Similarity')
        plt.xlabel('Embedding Index')
        plt.ylabel('Embedding Index')
    plt.tight_layout()
    save_path = os.path.join('plots', 'embedding_cosine_similarity.png')
    plt.savefig(save_path)
    plt.close()
