import matplotlib.pyplot as plt
import os

def plot_class_distribution(loader, dataset, title, ax):
    
    # Initialize empty list to store class labels
    class_labels = []

    # Iterate over the loader to extract class labels
    for _, _, _, anchor_label, _, _ in loader:
        class_labels.extend(anchor_label.numpy().tolist())

    # Plot histogram of class distribution
    ax.hist(class_labels, bins=len(dataset.classes), color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Frequency')
    ax.set_xticks(range(len(dataset.classes)))
    ax.set_xticklabels(dataset.classes, rotation=45, ha='right')

def plot_class_distributions(train_loader_triplet, val_loader_triplet, full_dataset):
    # Create a figure for plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot training dataset distribution
    plot_class_distribution(train_loader_triplet, full_dataset, 'Training Dataset Class Distribution', axes[0])

    # Plot validation dataset distribution
    plot_class_distribution(val_loader_triplet, full_dataset, 'Validation Dataset Class Distribution', axes[1])

    # Adjust layout and display plot
    fig.tight_layout()
    save_path = os.path.join('plots','dataset_class_distributions.png')
    plt.savefig(save_path)
    plt.close()

def plot_batch_triplet(train_loader_triplet, num_images=5):
    # Define label names for FashionMNIST
    class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    for batch in train_loader_triplet:
        # Extract images and labels from the batch
        anchor_images = batch[0]
        positive_images = batch[1]
        negative_images = batch[2]
        anchor_labels = batch[3]  # Assuming the class labels are provided in the batch
        positive_labels = batch[4]
        negative_labels = batch[5]

        # Plot several images in subplots
        fig, axes = plt.subplots(3, num_images, figsize=(12, 6))

        # Plot each image with its corresponding class label as the title
        for i in range(num_images):
            # Anchor image
            ax = axes[0, i]
            ax.imshow(anchor_images[i].permute(1, 2, 0).squeeze(), cmap='gray')
            ax.axis('off')  # Hide axis
            ax.set_title(f'Anchor - {class_labels[anchor_labels[i]]}')

            # Positive image
            ax = axes[1, i]
            ax.imshow(positive_images[i].permute(1, 2, 0).squeeze(), cmap='gray')
            ax.axis('off')  # Hide axis
            ax.set_title(f'Pos - {class_labels[positive_labels[i]]}')

            # Negative image
            ax = axes[2, i]
            ax.imshow(negative_images[i].permute(1, 2, 0).squeeze(), cmap='gray')
            ax.axis('off')  # Hide axis
            ax.set_title(f'Neg - {class_labels[negative_labels[i]]}')

        fig.suptitle('Batch Triplet Images with Anchor, Positive, and Negative Labels', fontsize=16)
        save_path = os.path.join('plots','batch_triplet.png')
        plt.savefig(save_path)
        plt.close()
        break  # Only print the first batch
