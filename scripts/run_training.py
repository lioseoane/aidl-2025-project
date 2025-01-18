from src.data.dataloader import create_dataloaders
from src.data.load_mpii_data import load_mpii_data
from src.models.dummy_cnn import KeypointModel
from src.training.train import train_model
import torch
import torch.optim as optim

# Load data (images and keypoints)
keypoints_array, images_array, head_boxes_array = load_mpii_data()

# Create dataloaders
train_loader, val_loader = create_dataloaders(images_array, keypoints_array, batch_size=32, resize_to=128)

# Initialize model, optimizer, and loss function
model = KeypointModel(num_keypoints=16, input_size=128)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# Train the model
train_model(train_loader, model, optimizer, criterion, num_epochs=2)

# Save the trained model
model_save_path = './keypoint_model.pth'  # Specify the path to save the model
torch.save(model.state_dict(), model_save_path)