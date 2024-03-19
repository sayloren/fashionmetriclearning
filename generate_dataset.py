from torch.utils.data import Dataset
import random
import torch
from torchvision.transforms import ToPILImage


class TripletMNIST(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        anchor_img, anchor_label = self.dataset[index]
        
        # Choose positive sample with the same label as anchor
        positive_index = index
        while positive_index == index:
            positive_index = random.randint(0, len(self.dataset) - 1)
        positive_img, positive_label = self.dataset[positive_index]

        # Ensure the positive sample has the same label as the anchor
        while positive_label != anchor_label:
            positive_index = random.randint(0, len(self.dataset) - 1)
            positive_img, positive_label = self.dataset[positive_index]

        # Choose negative sample with different label from anchor
        negative_index = index
        while negative_index == index or self.dataset[negative_index][1] == anchor_label:
            negative_index = random.randint(0, len(self.dataset) - 1)
        negative_img, negative_label = self.dataset[negative_index]

        return anchor_img, positive_img, negative_img, anchor_label, positive_label, negative_label
    
def subset_transform(dataset, transform):
    to_pil = ToPILImage()
    transformed_data = []
    for idx in range(len(dataset)):
        item, label = dataset[idx]
        item_pil = to_pil(item)  # Convert tensor to PIL Image
        transformed_item = transform(item_pil)
        transformed_data.append((transformed_item, label))
    return transformed_data