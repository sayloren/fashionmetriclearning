import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import os
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.calibration import calibration_curve

def plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    # Plot the loss and accuracy curves
    plt.figure(figsize=(10, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curves')
    plt.legend()
    save_path = os.path.join('plots','loss_and_accuracy_curves.png')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(matched_model, val_loader_triplet):
    # Define label names for FashionMNIST
    label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Initialize lists to store true labels and predicted scores
    true_labels = []
    predicted_scores = []

    # Iterate through the data loader to get predictions
    with torch.no_grad():
        for batch in val_loader_triplet:
            anchor_imgs, positive_imgs, negative_imgs, anchor_labels, _, _ = batch
            output = matched_model.forward_once(anchor_imgs)
            true_labels.extend(anchor_labels.numpy())
            predicted_scores.extend(output.numpy())

    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_scores = np.array(predicted_scores)

    # Calculate the number of classes
    num_classes = len(np.unique(true_labels))

    # Plot ROC curves for each class
    plt.figure(figsize=(10, 8))
    for class_label in range(num_classes):
        # Extract true labels and predicted scores for the current class
        class_true_labels = (true_labels == class_label).astype(int)
        class_predicted_scores = predicted_scores[:, class_label]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(class_true_labels, class_predicted_scores)

        # Calculate AUROC score
        auroc = roc_auc_score(class_true_labels, class_predicted_scores)

        # Plot ROC curve with label names
        plt.plot(fpr, tpr, label=f'{label_names[class_label]} (AUROC = {auroc:.2f})')

    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

    # Set plot labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for FashionMNIST Classes')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join('plots','roc.png')
    plt.savefig(save_path)
    plt.close()

# Confusion matrix, just in case, but not used in metric learning model
def plot_confusion_matrix(loader, model):
    # Define label names for FashionMNIST
    label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Collect true labels and predicted labels
    true_labels = []
    predicted_labels = []
    min_true_label = float('inf')  # Initialize with a large value

    with torch.no_grad():
        for batch in loader:
            anchor_imgs, _, _ = batch[:3]  # Only need anchor images
            outputs = model.forward_once(anchor_imgs)
            
            # Convert predicted labels to the same range as true labels
            predicted_labels.extend(torch.argmax(outputs, axis=1).cpu().numpy())
            # print(predicted_labels)
            true_labels.extend(batch[3].cpu().numpy())  # Assuming the anchor labels are at index 3
            # print(true_labels)
            min_true_label = min(min_true_label, min(true_labels))

    # Align predicted labels to the same range as true labels
    predicted_labels = [label - min_true_label for label in predicted_labels]

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, square=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for FashionMNIST Triplet Loss Model')
    
    save_path = os.path.join('plots','confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()

def error_analysis(model, test_loader):
    # Step 2: Make predictions
    all_embeddings = []
    all_true_labels = []

    with torch.no_grad():
        for anchor_images, positive_images, negative_images, labels, _, _ in test_loader:
            embeddings = model(anchor_images, positive_images, negative_images)  # Get embeddings from the model
            # If the model returns multiple outputs as a tuple, use only the first one
            embeddings = embeddings[0] if isinstance(embeddings, tuple) else embeddings
            all_embeddings.extend(embeddings.tolist())
            all_true_labels.extend(labels.tolist())

    # Step 3: Identify misclassified examples
    predictions = [torch.argmax(torch.tensor(embedding)) for embedding in all_embeddings]
    misclassified_indices = [i for i, (pred, true) in enumerate(zip(predictions, all_true_labels)) if pred != true]

    # Step 4: Visualize misclassified examples
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for i, idx in enumerate(misclassified_indices[:5]):  # Visualize the first 5 misclassified examples
        image, _, _, true_label, _, _ = test_loader.dataset[idx]
        predicted_label = predictions[idx]
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f'Predicted: {predicted_label}, True: {true_label}')
        axes[i].axis('off')

    fig.suptitle('Error Prone Images')
    plt.tight_layout()
    save_path = os.path.join('plots', 'error_analysis.png')
    plt.savefig(save_path)
    plt.close()

# Doesn't really mean anything for a metric learning model
def plot_calibration_curve(model, dataloader, positive_class=1, num_bins=10):
    y_true = []
    y_prob = []

    # Predict probabilities and collect true labels
    for anchor_images, positive_images, negative_images, labels, _, _ in dataloader:
        with torch.no_grad():
            outputs1, _, _ = model(anchor_images, positive_images, negative_images)
            probabilities = torch.softmax(outputs1, dim=1)
            # Convert multiclass labels to binary labels
            labels_binary = (labels == positive_class).numpy().astype(int)  # Convert to 1 for positive class, 0 otherwise
            y_true.extend(labels_binary)
            y_prob.extend(probabilities[:, positive_class].numpy())  # Probability of positive class

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=num_bins)

    # Plot calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of true positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join('plots', 'calibration_curve.png')
    plt.savefig(save_path)
    plt.close()