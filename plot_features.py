import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import math
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

def extract_and_visualize_feature_maps(model, input_image):
    def get_feature_maps(model, input_image):
        feature_maps = []
        x = input_image.unsqueeze(0)
        for layer in model.conv_layers:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps

    # Get the feature maps for the input image
    feature_maps = get_feature_maps(model, input_image)

    # Determine the layout of subplots
    num_layers = len(feature_maps)
    num_cols = 4
    num_rows = math.ceil((num_layers + 1) / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))

    # Plot the original image
    original_image = input_image.squeeze().detach().cpu().numpy()
    axs[0, 0].imshow(original_image, cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Original Image')

    for i, fmap in enumerate(feature_maps, start=1):
        fmap = fmap.squeeze().detach().cpu().numpy()
        if len(fmap.shape) == 3:
            fmap = fmap.mean(axis=0)  # Take the mean over channels
        row = i // num_cols
        col = i % num_cols
        axs[row, col].imshow(fmap, cmap='gray')
        axs[row, col].axis('off')
        axs[row, col].set_title(f'Layer {i}')

    for i in range(num_layers + 1, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].axis('off')

    plt.suptitle('Feature Maps for One Image Through the Network')
    plt.tight_layout()
    save_path = os.path.join('plots', 'feature_maps.png')
    plt.savefig(save_path)
    plt.close()

import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the loss function for activation maximization
def activation_maximization_loss(output, target_neuron_index):
    # Negative mean activation of the target neuron to maximize its activation
    return -output[:, target_neuron_index].mean()

# Define the function to perform activation maximization
def activation_maximization(model, target_neuron_index, anchor_image, positive_image, negative_image, lr, num_iterations=1000):
    # Initialize the optimizer for all three input images
    optimizer = optim.SGD([anchor_image.requires_grad_(True), positive_image.requires_grad_(True), negative_image.requires_grad_(True)], lr=lr)
    
    for i in range(num_iterations):
        # Forward pass through the model
        output1, _, _ = model(anchor_image.unsqueeze(0), positive_image.unsqueeze(0), negative_image.unsqueeze(0))
        
        # Compute the loss
        loss = activation_maximization_loss(output1, target_neuron_index)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"Iteration [{i+1}/{num_iterations}], Loss: {loss.item()}")
    
    # Convert the input images to numpy arrays for visualization
    optimized_anchor_image = anchor_image.detach().cpu().numpy()
    optimized_positive_image = positive_image.detach().cpu().numpy()
    optimized_negative_image = negative_image.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(optimized_anchor_image.squeeze(), cmap='gray')
    axes[0].axis('off')
    axes[0].set_title(f'Optimized Anchor Image Layer {target_neuron_index}')

    axes[1].imshow(optimized_negative_image.squeeze(), cmap='gray')
    axes[1].axis('off')
    axes[1].set_title(f'Optimized Negative Image Layer {target_neuron_index}')

    fig.suptitle('Activation Maximization for Two Classes')
    plt.tight_layout()
    save_path = os.path.join('plots', 'activation_maximized.png')
    plt.savefig(save_path)
    plt.close()

def visualize_filters(model):
    # Get the convolutional layers
    conv_layers = model.conv_layers

    # Extract the weights (filters) of the convolutional layers
    filters = []
    for layer in conv_layers:
        if isinstance(layer, nn.Conv2d):
            filters.append(layer.weight.data)

    # Plot the filters
    _, axs = plt.subplots(len(filters), 5, figsize=(12, 4 * len(filters)))
    for i, filter_bank in enumerate(filters):
        num_filters = min(5, filter_bank.size(0))
        filter_bank = filter_bank.detach().cpu()
        for j in range(num_filters):
            axs[i, j].imshow(filter_bank[j, 0], cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title(f'Layer {i+1}, Filter {j+1}')

    # Hide unused subplots
    for i in range(len(filters)):
        for j in range(num_filters, 5):
            axs[i, j].axis('off')

    plt.tight_layout()
    save_path = os.path.join('plots', 'visualize_filters.png')
    plt.savefig(save_path)
    plt.close()

def get_feature_maps(model, input_image):
    feature_maps = []
    x = input_image.unsqueeze(0)
    for layer in model.conv_layers:
        x = layer(x)
        feature_maps.append(x)
    return feature_maps

def visualize_feature_maps(matched_model, triplet_dataset, scale_factor=5):    
    # Assuming model is your MatchedNetwork model
    # Assuming dataset is your TripletMNIST dataset
    for anchor_img, _, _, _, _, _ in triplet_dataset:
        # Pass the anchor image through the model
        feature_maps = get_feature_maps(matched_model, anchor_img)

        # Visualize feature maps for each layer
        for i, fmap in enumerate(feature_maps):
            num_features = fmap.size(1)
            num_rows = 1
            num_cols = num_features
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 6))  # Larger subplot size
            for j in range(num_features):
                # Upscale the feature map
                scaled_fmap = torch.nn.functional.interpolate(fmap[:, j].unsqueeze(0), scale_factor=scale_factor, mode='nearest')
                scaled_image = scaled_fmap[0].detach().cpu().numpy()
                scaled_image = np.squeeze(scaled_image)  # Squeeze to remove single-dimensional entries
                axs[j].imshow(scaled_image, cmap='gray', interpolation='nearest')  # Interpolation for smoother images
                axs[j].axis('off')
                if j == 0:
                    axs[j].set_title(f'Layer {i+1}')
            
            # Save subplot to a file
            save_path = os.path.join('plots', f'feature_maps_layer_{i+1}.png')
            plt.savefig(save_path, bbox_inches='tight')  # Adjust bounding box for tight layout
            plt.close()
        break  # Remove this line if you want to visualize feature maps for all anchor images





# # Not yet working
# def activation_maximization_per_class(model, target_layer, num_classes, unique_class_images, lr, num_iterations=1000, save_path=None):
#     fig, axs = plt.subplots(num_classes // 5, 5, figsize=(15, 3 * (num_classes // 5)))

#     for class_idx, class_image in unique_class_images.items():
#         # Convert the class image to a PyTorch tensor and add batch dimension
#         input_image = torch.tensor(class_image).unsqueeze(0).unsqueeze(0).float().requires_grad_(True)

#         # Set the model to evaluation mode
#         model.eval()

#         # Define the optimizer
#         optimizer = optim.SGD([input_image], lr=lr)

#         for i in range(num_iterations):
#             optimizer.zero_grad()

#             # Forward pass through the model
#             output1, _, _ = model(input_image)

#             # Calculate loss as the activation of the target neuron
#             loss = -output1[:, class_idx].mean()

#             # Backward pass
#             loss.backward()
#             optimizer.step()

#             # Clipping to maintain values in a reasonable range
#             input_image.data = torch.clamp(input_image.data, min=0)

#             if (i + 1) % 100 == 0:
#                 print(f"Iteration [{i+1}/{num_iterations}], Class: {class_idx}, Loss: {loss.item()}")

#         ax = axs[class_idx // 5, class_idx % 5]
#         ax.imshow(input_image.squeeze().detach().cpu().numpy(), cmap='gray')
#         ax.set_title(f"Class {class_idx}")
#         ax.axis('off')

#     plt.tight_layout()
#     save_path = os.path.join('plots', 'activation_maximized_by_class.png')
#     plt.savefig(save_path)
#     plt.close()

# # Not yet working
# def get_unique_class_images(train_loader_triplet, num_classes):
#     unique_images = {}
#     class_count = 0

#     for batch in train_loader_triplet:
#         for image, label in zip(batch[:3], batch[3]):
#             image = image.squeeze().detach().cpu().numpy()
#             if image.shape[0] == 64:
#                 image = image[0]  # Select the first image if multiple images are present
#             if label.item() not in unique_images:
#                 unique_images[label.item()] = image
#                 class_count += 1
#             if class_count == num_classes:
#                 break
#         if class_count == num_classes:
#             break

#     # Visualize the unique class images
#     fig, axs = plt.subplots(1, num_classes, figsize=(15, 3))
#     for class_idx, (label, image) in enumerate(unique_images.items()):
#         axs[class_idx].imshow(image, cmap='gray')
#         axs[class_idx].set_title(f"Class {label}")
#         axs[class_idx].axis('off')
#     plt.tight_layout()
#     save_path = os.path.join('plots', 'unique_classes.png')
#     plt.savefig(save_path)
#     plt.close()
#     return unique_images