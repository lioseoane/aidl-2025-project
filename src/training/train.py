import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataloader import create_dataloaders  # Import the function to create dataloaders
from src.data.load_mpii_data import load_mpii_data  # Assuming this loads the MPII dataset
from src.models.dummy_cnn import KeypointModel
import os
from tqdm import tqdm

def train_model(train_loader, model, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for batch_idx, (images, keypoints) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()  # Zero the gradients before backward pass

            # Forward pass
            outputs = model(images)  # Get predictions from the model
            loss = criterion(outputs, keypoints.view(-1, 32))  # assuming keypoints are of size 32 (x, y for each keypoint)

            # Backward pass and optimization
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model weights

            running_loss += loss.item()  # Accumulate loss for averaging
        
        # Print the loss at the end of each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")