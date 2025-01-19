import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataloader import create_dataloaders
from src.data.load_mpii_data import load_mpii_data
from src.models.dummy_cnn import KeypointModel
from src.training.evaluate import evaluate_model  # Import the evaluation function
import torch

# Load data (images and keypoints)
keypoints_array, images_array, head_boxes_array = load_mpii_data()

# Create dataloaders
train_loader, val_loader = create_dataloaders(images_array, keypoints_array, batch_size=32, resize_to=128)

# Initialize model
model = KeypointModel(num_keypoints=16, input_size=128)

# Load the trained model from a checkpoint
model_checkpoint_path = './checkpoints/model_epoch_10.pth'
model.load_state_dict(torch.load(model_checkpoint_path))
model.eval()  # Set the model to evaluation mode

# Define the loss function
criterion = torch.nn.MSELoss()

# Run the evaluation
evaluate_model(val_loader, model, criterion, log_dir="logs/val_logs")