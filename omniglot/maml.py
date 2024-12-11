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
def load_omniglot_data(root, background=True):
    dataset = Omniglot(root=root, background=background, download=True)
    class_to_images = {}
    for img, target in dataset:
        if target not in class_to_images:
            class_to_images[target] = []
        class_to_images[target].append(img)
    return class_to_images

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
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
    
    support_images = transform_images(support_imgs).to(device)
    query_images = transform_images(query_imgs).to(device)
    support_labels = torch.tensor(support_labels, dtype=torch.long).to(device)
    query_labels = torch.tensor(query_labels, dtype=torch.long).to(device)
    
    return support_images, support_labels, query_images, query_labels

#-----------------------------------
# Model Definition (Simple 4-layer CNN)
#-----------------------------------
class ConvNet(nn.Module):
    def __init__(self, num_classes=5):
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
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        logits = self.fc(features)
        return logits

#-----------------------------------
# Utilities for MAML
#-----------------------------------
from torch.nn.utils.stateless import functional_call

def clone_parameters(model):
    # Create a copy of model parameters as a dictionary
    return {name: p.clone() for name, p in model.named_parameters()}

def update_parameters(model, loss, params, lr):
    grads = torch.autograd.grad(loss, params.values(), create_graph=True)
    updated_params = {}
    for (name, param), grad in zip(params.items(), grads):
        updated_params[name] = param - lr * grad
    return updated_params

#-----------------------------------
# Meta-Training with MAML
#-----------------------------------
def meta_train_maml(root, meta_iterations=1000, inner_steps=5, inner_lr=0.01, meta_lr=1e-3, N=5, K=1, Q=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class_to_images_train = load_omniglot_data(root, background=True)
    model = ConvNet(num_classes=N).to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for iteration in range(meta_iterations):
        meta_optimizer.zero_grad()
        
        # Sample a task
        support_images, support_labels, query_images, query_labels = get_few_shot_task(
            class_to_images_train, N=N, K=K, Q=Q, device=device)
        
        # Clone model parameters
        fast_weights = clone_parameters(model)
        
        # Inner-loop adaptation on support set
        for _ in range(inner_steps):
            support_logits = functional_call(model, fast_weights, (support_images,))
            support_loss = loss_fn(support_logits, support_labels)
            fast_weights = update_parameters(model, support_loss, fast_weights, inner_lr)
        
        # Compute loss on query set with adapted parameters
        query_logits = functional_call(model, fast_weights, (query_images,))
        query_loss = loss_fn(query_logits, query_labels)
        
        # Meta-update
        query_loss.backward()
        meta_optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Query Loss: {query_loss.item():.4f}")

    print("Meta-training completed.")
    return model

#-----------------------------------
# Test-Time Adaptation (Evaluation)
#-----------------------------------
def evaluate_model_maml(model, root, N=5, K=1, Q=15, inner_steps=50, inner_lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    class_to_images_test = load_omniglot_data(root, background=False)
    loss_fn = nn.CrossEntropyLoss()
    
    # Sample a task from test set
    support_images, support_labels, query_images, query_labels = get_few_shot_task(
        class_to_images_test, N=N, K=K, Q=Q, device=device)
    
    # Adapt on support set
    fast_weights = clone_parameters(model)
    for _ in range(inner_steps):
        support_logits = functional_call(model, fast_weights, (support_images,))
        support_loss = loss_fn(support_logits, support_labels)
        grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=False)
        updated_params = {}
        for (name, param), grad in zip(fast_weights.items(), grads):
            updated_params[name] = param - inner_lr * grad
        fast_weights = updated_params
    
    # Evaluate on query set
    with torch.no_grad():
        query_logits = functional_call(model, fast_weights, (query_images,))
        pred_labels = torch.argmax(query_logits, dim=1)
        accuracy = (pred_labels == query_labels).float().mean().item()
    
    print(f"Test-Time Accuracy after adaptation: {accuracy * 100:.2f}%")


#-----------------------------------
# Example usage (uncomment to run):
#-----------------------------------
# root = "./omniglot_data"
# model = meta_train(root, meta_iterations=1000, N=5, K=1, Q=15)
# evaluate_model(model, root, N=5, K=1, Q=15)
