# my packages
from matched_network import MatchedNetwork
from generate_dataset import TripletMNIST, subset_transform
from plot_dataset import plot_class_distributions, plot_batch_triplet
from plot_performance import plot_roc_curves, plot_loss_and_accuracy, error_analysis 
from plot_embeddings import plot_embeddings_comparison, visualize_embeddings_with_images, plot_pairwise_distance_histogram, plot_pairwise_distance_histogram_by_class, plot_class_embeddings_similarity
from plot_features import extract_and_visualize_feature_maps, activation_maximization, visualize_filters, visualize_feature_maps
from plot_interpretability import generate_gradcam_overlay

# torch packages
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
import torch
from torch.utils.data import DataLoader, random_split # , WeightedRandomSampler
from torchvision import transforms

# other packages
import matplotlib.pyplot as plt
import argparse
# import random
import numpy as np
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script for Matched Network')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--subset_size', type=int, help='Size of the subset dataset, if left blank will run the whole dataset')
    parser.add_argument('--num_epochs',type=int, default=10, help='number of epochs to train the model')
    parser.add_argument('--patience', type=int, default=2, help='patience before early stopping')
    parser.add_argument('--labels_to_display', type=list, default = [5,9], help='a list of the clusters to visualize the images on')
    parser.add_argument('--load_model', type=str, default='false', help='Load model from checkpoint (default: false), if true will load from the last checkpoint file')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Initialize learning rate, Matched network, criterion, and optimizer
    learning_rate = args.learning_rate
    matched_model = MatchedNetwork()
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(matched_model.parameters(), lr=learning_rate)

    # Load FashionMNIST dataset
    print('Loading the fashionMNIST dataset')
    full_dataset = FashionMNIST(root='./data', train=True, transform=transforms.ToTensor()) # download = True

    # # Sampler doesn't work yet 
    # # Count the occurrences of each class in the dataset
    # class_counts = torch.zeros(len(full_dataset.classes))
    # for _, label in full_dataset:
    #     class_counts[label] += 1

    # # Compute total number of samples
    # total_samples = sum(class_counts)

    # # Compute class weights for balancing
    # class_weights = total_samples / class_counts

    # # Print out the class weights
    # print("Class Weights:")
    # for i, weight in enumerate(class_weights):
    #     print(f"Class {i}: {weight.item()}")

    # # Define sampler for class balancing
    # class_sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(full_dataset), replacement=True)

    # Choose the size of the subset
    print('Subsetting the dataset')
    if args.subset_size:
        subset_size =  args.subset_size
    else:
        subset_size =  len(full_dataset)

    # Randomly select indices for the subset
    subset_indices = torch.randperm(len(full_dataset))[:subset_size]

    # Create a subset of the dataset
    subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)

    # Apply data augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    # Apply the transformation to the subset dataset
    print('Applying transforms')
    augmented_subset = subset_transform(subset_dataset, transform)

    # Create triplet dataset
    print('Creating the dataset')
    triplet_dataset = TripletMNIST(subset_dataset) # augmented_subset

    # Split dataset into train and validation sets
    print('Creating training and validation split')
    train_size = int(0.8 * len(triplet_dataset))
    val_size = len(triplet_dataset) - train_size
    train_subset, val_subset = random_split(triplet_dataset, [train_size, val_size])

    # Create data loaders for the train and validation sets with triplets
    print('Creating the dataloaders')
    train_loader_triplet = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader_triplet = DataLoader(val_subset, batch_size=64, shuffle=False)

    # Plot the distributions for the training and validation dataset classes
    print('Plotting the class distributions')
    plot_class_distributions(train_loader_triplet, val_loader_triplet, full_dataset)

    # Plot a batch of anchors, positives, and negatives
    print('Plotting the anchors positives and negatives')
    plot_batch_triplet(train_loader_triplet, num_images=5)
    
    # Initalize the checkpoints directory
    checkpoint_dir = 'checkpoints'


    # Make it so that I can continue training from the checkpoints, as well as rpinting loss and rocs
    
    if args.load_model.lower() == 'true':
        print('Loading from last checkpoint')
        # Function to extract the epoch number from the checkpoint filename
        def get_epoch_from_checkpoint(checkpoint_filename):
            return int(checkpoint_filename.split('_')[-1].split('.')[0])

        # Get list of checkpoint files in the directory
        checkpoint_files = os.listdir(checkpoint_dir)
        checkpoint_files = [f for f in checkpoint_files if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        
        # Sort checkpoint files based on epoch number
        checkpoint_files.sort(key=lambda x: get_epoch_from_checkpoint(x))

        # Select the latest checkpoint file
        latest_checkpoint_file = os.path.join(checkpoint_dir, checkpoint_files[-1])

        # Load the model from the latest checkpoint file
        checkpoint = torch.load(latest_checkpoint_file)
        matched_model.load_state_dict(checkpoint['model_state_dict'])

        # Set the model to evaluation mode
        matched_model.eval()

    else:
        print('Starting training')
        # Set the hyperparameters 
        num_epochs = args.num_epochs
        best_val_loss = np.inf
        patience = args.patience

        # Set the directory to save the checkpoints
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize the loss and accuracy captures
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            # Train the model
            matched_model.train()
            total_train_loss = 0.0
            correct_train = 0
            total_train = 0
            for batch in train_loader_triplet:
                optimizer.zero_grad()
                anchor_imgs, positive_imgs, negative_imgs = batch[:3]
                output1, output2, output3 = matched_model(anchor_imgs, positive_imgs, negative_imgs)
                loss = criterion(output1, output2, output3)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                
                # Compute training accuracy
                _, predicted = torch.max(output1, 1)  # Assuming output1 is the output of the model
                total_train += anchor_imgs.size(0)
                correct_train += (predicted == batch[3]).sum().item()  # Assuming true labels are at index 3 in the batch
            
            # collect the training loss and accuracies
            avg_train_loss = total_train_loss / len(train_loader_triplet)
            train_losses.append(avg_train_loss)
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)

            # Validation loop
            matched_model.eval()
            total_val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for batch in val_loader_triplet:
                    anchor_imgs, positive_imgs, negative_imgs = batch[:3]
                    output1, output2, output3 = matched_model(anchor_imgs, positive_imgs, negative_imgs)
                    loss = criterion(output1, output2, output3)
                    total_val_loss += loss.item()
                    # Compute validation accuracy
                    _, predicted = torch.max(output1, 1)  # Assuming output1 is the output of the model
                    total_val += anchor_imgs.size(0)
                    correct_val += (predicted == batch[3]).sum().item()  # Assuming true labels are at index 3 in the batch
            
            # collect the validation loss and accuracies
            avg_val_loss = total_val_loss / len(val_loader_triplet)
            val_losses.append(avg_val_loss)
            val_accuracy = correct_val / total_val
            val_accuracies.append(val_accuracy)

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {round(avg_train_loss,4)}, Validation Loss: {round(avg_val_loss,4)}, Training Accuracy: {round(train_accuracy,4)}, Validation Accuracy: {round(val_accuracy,4)}")

            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"Validation loss did not improve for {patience} epochs. Early stopping...")
                    break
                    
            # Save model checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': matched_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses,
                'train_accuracy': train_accuracies,
                'val_accuracy': val_accuracies,
            }, checkpoint_path)

        print('Finished Training')
        # Save the model weights
        torch.save(matched_model.state_dict(), 'model_weights.pth')

        # Plot the loss and accuracy curves
        print('Plotting the loss and accuracy curves')
        plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)

        # Plot the roc for each class
        print('Plotting the roc for each class')
        plot_roc_curves(matched_model, val_loader_triplet)

    # Plot the embeddings for the training and test
    print('Plotting the embeddings')
    validation_embeddings = plot_embeddings_comparison(matched_model, train_loader_triplet, val_loader_triplet)

    # Plot the images on the embedding graph for a subset of the classes in the validation set
    labels_to_display = args.labels_to_display
    print('Plotting the embeddings for classes {0}'.format(labels_to_display))
    visualize_embeddings_with_images(val_loader_triplet, matched_model, labels_to_display)

    # Assuming label_names is defined as ['T-shirt/top', 'Trouser', ...]
    print('Plotting the pairwise distances')
    plot_pairwise_distance_histogram(validation_embeddings)

    # Plot the pairwise distances by class for the validation embeddings
    print('Plotting the pairwise distances by class')
    plot_pairwise_distance_histogram_by_class(validation_embeddings)

    # Plot the cosine similiarites between embeddings per each class
    print('Plotting the cosine similarity between embeddings in each class')
    plot_class_embeddings_similarity(triplet_dataset, matched_model)

    # Plot the top 5 images the model struggles with - makes errors in the embedding space
    print('Plotting the error analysis')
    error_analysis(matched_model, val_loader_triplet)

    for batch in train_loader_triplet:
        # Extract images and labels from the batch
        anchor_images = batch[0]
        positive_images = batch[1]
        negative_images = batch[2]

    # Plot the feature maps
    print('Plotting the feature map through the model for one image')
    extract_and_visualize_feature_maps(matched_model, anchor_images[0])
    visualize_feature_maps(matched_model, triplet_dataset)

    # Plot the grad cam of a couple images
    print('Plotting grad cam')
    generate_gradcam_overlay(matched_model, triplet_dataset)

    # Visuzalize 5 filters for each layer
    print('Visualizing filters')
    visualize_filters(matched_model)

    # Plot the activation maximization for the anchor and negative image
    print('Plotting the activation maximization for the anchor and negative image classes')
    target_neuron_index = 0  # Choose the index of the target neuron
    activation_maximization(matched_model, target_neuron_index, anchor_images[0], positive_images[0], negative_images[0], learning_rate)

    # # These sections are still not working

    # # Plot an example of each unique class
    # print('Plotting unique classes example')
    # # Assuming train_loader_triplet is your DataLoader and dataset is the dataset used in it
    # class_labels = []
    # for batch in train_loader_triplet:
    #     anchor_labels = batch[3]  # Assuming true labels are at index 3 in the batch
    #     class_labels.extend(anchor_labels.tolist())
    # # Get unique class labels
    # unique_class_labels = set(class_labels)
    # num_classes = len(unique_class_labels)
    # unique_images = get_unique_class_images(train_loader_triplet, num_classes)

    # # Visualizing the activation maximization for each class
    # print('Plotting the activation maximization for each class')
    # target_layer = 0  # Change it to the index of the layer you're interested in
    # # Perform activation maximization for each class
    # activation_maximization_per_class(matched_model, target_layer, num_classes, unique_images, learning_rate)



if __name__ == "__main__":
    main()

