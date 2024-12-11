import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Omniglot
from PIL import Image
import numpy as np
import os
import random

#-----------------------------------
# Dataset Preparation (Omniglot)
#-----------------------------------
# We will create a dictionary mapping class_index -> list_of_images (PIL)
# Omniglot dataset from torchvision:
# - background=True: training set (30 alphabets)
# - background=False: evaluation set (20 alphabets)
# Each sample is (image, target) where target is the class index.
def load_omniglot_data(root, background=True):
    dataset = Omniglot(root=root, background=background, download=True)
    # Group images by class
    class_to_images = {}
    for img, target in dataset:
        if target not in class_to_images:
            class_to_images[target] = []
        class_to_images[target].append(img)
    return class_to_images

# A transform for Omniglot images
# Omniglot images are grayscale (1 channel), original size ~105x105
# Common practice: resize to 28x28 and just convert to tensor.
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # No normalization needed, but could add if desired.
    # transforms.Normalize((0.92206,), (0.08426,))
])

def transform_images(image_list):
    images = []
    for img in image_list:
        img = img.convert('L')  # ensure grayscale
        img = transform(img)    # (1, 28, 28)
        images.append(img)
    return torch.stack(images, dim=0)  # (B, 1, H, W)

#-----------------------------------
# Few-Shot Task Sampling for Omniglot
#-----------------------------------
def get_few_shot_task(class_to_images, N=5, K=1, Q=15, device='cuda'):
    """
    Given a dictionary mapping class_index -> list_of_PIL_images,
    Sample an N-way K-shot Q-query task.
    
    Returns:
        support_images: (N*K, 1, 28, 28) tensor
        support_labels: (N*K,) tensor of labels in [0, N-1]
        query_images: (N*Q, 1, 28, 28) tensor
        query_labels: (N*Q,) tensor of labels in [0, N-1]
    """
    # Randomly choose N classes
    selected_classes = random.sample(list(class_to_images.keys()), N)
    
    support_imgs = []
    query_imgs = []
    support_labels = []
    query_labels = []
    
    for i, cls in enumerate(selected_classes):
        imgs = class_to_images[cls]
        random.shuffle(imgs)
        support = imgs[:K]
        query = imgs[K:K+Q]
        
        support_imgs.extend(support)
        query_imgs.extend(query)
        
        support_labels.extend([i]*K)
        query_labels.extend([i]*Q)
    
    # Transform PIL images to tensors
    support_images = transform_images(support_imgs).to(device)  # (N*K, 1, H, W)
    query_images = transform_images(query_imgs).to(device)      # (N*Q, 1, H, W)
    support_labels = torch.tensor(support_labels, dtype=torch.long).to(device)
    query_labels = torch.tensor(query_labels, dtype=torch.long).to(device)
    
    return support_images, support_labels, query_images, query_labels

#-----------------------------------
# Model Definition (Simple 4-layer CNN adapted for Omniglot)
#-----------------------------------
# Adjust input channels to 1 for grayscale images.
class ConvNet(nn.Module):
    def __init__(self, num_classes=5, memory_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveMaxPool2d(1)
        )
        self.fc = nn.Linear(64 + memory_dim, num_classes)

    def forward(self, x, memory):
        features = self.encoder(x)  
        features = features.view(features.size(0), -1)  
        memory_rep = memory.unsqueeze(0).expand(features.size(0), -1) 
        combined = torch.cat([features, memory_rep], dim=1)  
        logits = self.fc(combined)  
        return logits

#-----------------------------------
# Meta-Training
#-----------------------------------
def meta_train(root, meta_iterations=1000, inner_steps=5, adapt_lr=0.01, N=5, K=1, Q=15, memory_dim=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load training data (background=True)
    class_to_images_train = load_omniglot_data(root, background=True)
    
    model = ConvNet(num_classes=N, memory_dim=memory_dim).to(device)
    model_optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    for iteration in range(meta_iterations):
        # Sample a new few-shot classification task from training data
        support_images, support_labels, query_images, query_labels = get_few_shot_task(
            class_to_images_train, N=N, K=K, Q=Q, device=device)
        
        # Initialize memory for this task
        memory = torch.zeros(memory_dim, requires_grad=True, dtype=torch.float32, device=device)
        memory_optimizer = optim.SGD([memory], lr=adapt_lr)
        
        # Inner-loop adaptation: update memory to fit the support set
        for param in model.parameters():
            param.requires_grad = False
            
        for _ in range(inner_steps):
            memory_optimizer.zero_grad()
            logits = model(support_images, memory)
            loss = loss_fn(logits, support_labels)
            loss.backward()
            memory_optimizer.step()
        
        # Meta-update the model with the query set
        for param in model.parameters():
            param.requires_grad = True
            
        logits_q = model(query_images, memory)
        query_loss = loss_fn(logits_q, query_labels)
        
        model_optimizer.zero_grad()
        query_loss.backward()
        model_optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Query Loss: {query_loss.item():.4f}")

    print("Meta-training completed.")
    return model

#-----------------------------------
# Test-Time Adaptation (Evaluation)
#-----------------------------------
def evaluate_model(model, root, N=5, K=1, Q=15, adapt_lr=0.01, memory_dim=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load test data (background=False)
    class_to_images_test = load_omniglot_data(root, background=False)
    
    for param in model.parameters():
        param.requires_grad = False
        
    loss_fn = nn.CrossEntropyLoss()
    
    # Sample a new unseen task
    support_images, support_labels, query_images, query_labels = get_few_shot_task(
        class_to_images_test, N=N, K=K, Q=Q, device=device)
    
    # Adapt memory
    memory = torch.zeros(memory_dim, requires_grad=True, dtype=torch.float32, device=device)
    memory_optimizer = optim.SGD([memory], lr=adapt_lr)
    
    inner_steps = 50
    for _ in range(inner_steps):
        memory_optimizer.zero_grad()
        logits = model(support_images, memory)
        loss = loss_fn(logits, support_labels)
        loss.backward()
        memory_optimizer.step()
    
    # Evaluate on query set after adaptation
    with torch.no_grad():
        logits_q = model(query_images, memory)
        pred_labels = torch.argmax(logits_q, dim=1)
        accuracy = (pred_labels == query_labels).float().mean().item()
    
    print(f"Test-Time Accuracy after Memory Adaptation: {accuracy * 100:.2f}%")

#-----------------------------------
# Example usage:
# The dataset will be automatically downloaded at `root` directory.
#-----------------------------------
# root = "./omniglot_data"
# model = meta_train(root, meta_iterations=1000, N=5, K=1, Q=15)
# evaluate_model(model, root, N=5, K=1, Q=15)
