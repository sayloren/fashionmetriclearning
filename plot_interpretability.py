# from lime.lime_image import LimeImageExplainer
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import torch
import cv2
from skimage.color import gray2rgb
# import shap
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier

# Generally not used for metric learning
def lime_interpret(model, loader, image_index=0):
    # Convert validation data to numpy arrays
    val_data = []
    for batch in loader:
        anchor_imgs = batch  # Assuming the batch contains anchor positive and negative
        val_data.append(anchor_imgs.numpy())  # Convert anchor images to numpy arrays

    val_data = np.concatenate(val_data, axis=0)  # Concatenate batches into a single numpy array

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate explanations for a sample of the validation data
    explanation = explainer.explain_instance(val_data[image_index], model.predict, hide_color=0, top_labels=5, num_samples=1000)

    # Visualize the explanation
    fig, ax = plt.subplots(figsize=(6, 6))
    explanation.show_in_notebook(ax=ax)
    plt.tight_layout()
    save_path = os.path.join('plots','lime_interpretability.png')
    plt.savefig(save_path)
    plt.close()
        
# Generally not used for metric learning
def shap_interpreter(model, loader, num_samples=100):

    # Convert validation data to numpy arrays
    val_data = []
    for batch in loader:
        anchor_imgs = batch[0]  # Extract anchor images from the batch
        val_data.append(anchor_imgs.numpy())  # Convert anchor images to numpy arrays

    val_data = np.concatenate(val_data, axis=0) 

    # Initialize SHAP explainer
    explainer = shap.Explainer(model, val_data)

    # Compute SHAP values
    shap_values = explainer.shap_values(val_data)

    # Visualize SHAP values
    shap.summary_plot(shap_values, val_data_flat)

    # Plot the explanation
    plt.tight_layout()
    save_path = os.path.join('plots', 'shap_interpretability.png')
    plt.savefig(save_path)
    plt.close()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        self.feature_grad = None
        self.hook = self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            output.requires_grad_(True)

        def backward_hook(module, grad_in, grad_out):
            self.feature_grad = grad_out[0]

        hook = self.target_layer.register_forward_hook(forward_hook)
        _ = self.target_layer.register_backward_hook(backward_hook)
        return hook

    def generate_grad_cam(self, input_image1, input_image2, input_image3, target_class=None):
        self.model.zero_grad()
        output1, _, _ = self.model(input_image1, input_image2, input_image3)  # Forward pass through the Matched network
        if target_class is None:
            target_class = torch.argmax(output1, dim=1)

        # Compute gradients for the target class
        one_hot_output = torch.zeros_like(output1)
        one_hot_output[torch.arange(len(output1)), target_class] = 1
        output1.backward(gradient=one_hot_output, retain_graph=True)

        # Compute Grad-CAM
        weights = torch.mean(self.feature_grad, dim=(2, 3), keepdim=True)
        grad_cam = torch.mean(weights * self.feature_grad, dim=1).squeeze()
        grad_cam = F.relu(grad_cam)

        # Normalize Grad-CAM
        grad_cam = (grad_cam - torch.min(grad_cam)) / (torch.max(grad_cam) - torch.min(grad_cam) + 1e-8)

        return grad_cam.detach().cpu().numpy()

    def remove_hooks(self):
        self.hook.remove()

# Define a function to overlay the grad cam on the original image
def overlay_grad_cam(input_image, grad_cam_map, alpha=0.3):
    # Convert the input image to a numpy array and scale it to the range [0, 255]
    input_image_np = (input_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # Resize the Grad-CAM map to match the input image dimensions
    grad_cam_map_resized = cv2.resize(grad_cam_map, (input_image_np.shape[1], input_image_np.shape[0]))
    
    # Apply the colormap to the resized Grad-CAM map
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map_resized), cv2.COLORMAP_HOT)  # Using a bright colormap (e.g., HOT)
    
    # Convert the input image to a three-channel image
    input_image_resized = cv2.cvtColor(input_image_np, cv2.COLOR_GRAY2RGB)
    
    # Add the heatmap to the original image with transparency
    overlayed_image = cv2.addWeighted(input_image_resized, 1 - alpha, heatmap, alpha, 0)
  
    return overlayed_image

def generate_gradcam_overlay(model, dataset, conv_layer=-1, alpha=0.3, save_path="gradcam_subplots.png"):
    # Define label names for FashionMNIST
    class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Initialize GradCAM object
    grad_cam = GradCAM(model, model.conv_layers[conv_layer])  # Assuming last convolutional layer is the target layer

    # Create lists to store original images and overlaid Grad-CAM images
    original_images = []
    grad_cam_images = []
    original_labels = []

    # Iterate over the first 5 images in the dataset
    for i in range(5):
        anchor_img, positive_img, negative_img, anchor_label, positive_label, negative_label = dataset[i]

        # Pass the anchor, positive, and negative images to the GradCAM object to generate Grad-CAM
        grad_cam_map = grad_cam.generate_grad_cam(anchor_img.unsqueeze(0), positive_img.unsqueeze(0), negative_img.unsqueeze(0), target_class=anchor_label)
        
        # Overlay Grad-CAM on input image
        overlayed_image = overlay_grad_cam(anchor_img, grad_cam_map, alpha)
        
        # Append original image and overlaid Grad-CAM image to lists
        original_images.append(anchor_img.permute(1, 2, 0))
        grad_cam_images.append(overlayed_image)
        original_labels.append(anchor_label)

    # Create subplots for the tiled images
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))

    # Plot original images
    for i, original_image in enumerate(original_images):
        axs[0, i].imshow(original_image, cmap='gray')
        # axs[0, i].set_title(f'Image {i+1}')
        axs[0, i].set_title(class_labels[original_labels[i]])
        axs[0, i].axis('off')

    # Plot overlaid Grad-CAM images
    for i, grad_cam_image in enumerate(grad_cam_images):
        axs[1, i].imshow(grad_cam_image)
        axs[1, i].set_title(f'Grad-CAM Image {i+1}')
        axs[1, i].axis('off')
    
    # Save the figure
    fig.suptitle('GradCAM')
    plt.tight_layout()
    save_path = os.path.join('plots', 'gradcam.png')
    plt.savefig(save_path)
    plt.close()
    grad_cam.remove_hooks()
