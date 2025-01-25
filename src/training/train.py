import torch
from torch.utils.tensorboard import SummaryWriter  
from src.utils.metrics import calculate_pck
from src.utils.visualization import visualize_keypoints
import os
from tqdm import tqdm


def train_model(train_loader, model, optimizer, criterion, num_epochs=10, log_dir="logs/train_logs", checkpoint_dir="checkpoints"):

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    
    model = model.to(device)
    print(f"Using device: {device}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        total_pck = 0.0
        total_samples = 0

        #for images, targets in train_loader:
            #print(f"Batch size: {images.size(0)}")  # Prints batch size (number of images)
            #print(f"Targets length: {len(targets)}")  # Prints the length of targets
        
        for batch_idx, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):

            # Move data to the same device as the model
            # print(targets)
            images = images.to(device)

            # if len(targets) == 1:  # Dividir si targets contiene un único diccionario
            #     batch_targets = []
            #     for i in range(len(images)):  # Para cada imagen en el lote
            #         single_target = {
            #             'boxes': targets[0]['boxes'][i],  # Extraer cajas de la imagen i
            #             'labels': targets[0]['labels'][i],  # Extraer etiquetas de la imagen i
            #             'keypoints': targets[0]['keypoints'][i]  # Extraer puntos clave de la imagen i
            #         }
            #         batch_targets.append(single_target)  # Añadir a la lista de targets
            #     targets = batch_targets  # Reemplazar el target original con la lista dividida

            # print(f"DEBUG: Number of targets after splitting: {len(targets)}")

            # # Validar y mover targets al dispositivo
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


            # Reshape targets into a list of dictionaries for each image
            new_targets = []
            for i in range(len(images)):  # Assuming images is of size [batch_size, ...]
                target_dict = {
                    "boxes": targets["boxes"][i:i+1],  # Take only the boxes for the i-th image
                    "labels": targets["labels"][i:i+1],  # Take only the labels for the i-th image
                    "keypoints": targets["keypoints"][i:i+1],  # Take only the keypoints for the i-th image
                }
                new_targets.append(target_dict)

            # Move each target to the device
            for t in new_targets:
                for k, v in t.items():
                    t[k] = v.to(device)

            # targets = [{k: v.to(device) for k, v in targets.items()}]

            optimizer.zero_grad()  # Zero the gradients before backward pass

            # print(f"DEBUG: Number of images in batch: {len(images)}")
            # print(f"DEBUG: Number of targets in batch: {len(targets)}")
            # for i, t in enumerate(targets):
            #     print(f"Target {i}:")
            #     print(f"  Keys: {list(t.keys())}")
            #     for k, v in t.items():
            #         print(f"    Key: {k}, Value type: {type(v)}, Shape: {v.shape if isinstance(v, torch.Tensor) else 'N/A'}")

            # Forward pass
            #outputs = model(images, targets)  # Get predictions from the model
            outputs = model(images, new_targets)  # Get predictions from the model

            # Extract model outputs
            pred_boxes = outputs['boxes']  # Predicted bounding boxes
            pred_labels = outputs['labels']  # Predicted class labels
            pred_scores = outputs['scores']  # Predicted class scores
            pred_keypoints = outputs['keypoints']  # Predicted keypoints

             # Unpack targets (bbox, keypoints, class_name)
            target_bboxes = [target['boxes'] for target in targets]
            target_keypoints = [target['keypoints'] for target in targets]
            target_class_names = [target['class_name'] for target in targets]

            # Losses from the model (classification + bounding boxes + keypoints)
            loss_dict = criterion(outputs, targets)

            # Keypoint loss (part of the total loss)
            loss_keypoints = loss_dict['keypoints_loss']

            # loss = criterion(outputs, keypoints.view(-1, 32))  # assuming keypoints are of size 32 (x, y for each keypoint)

            # Calculate PCK (Percentage of Correct Keypoints)
            pred_keypoints_np = pred_keypoints.cpu().detach().numpy().reshape(-1, 16, 2)  # Reshape to keypoints format
            true_keypoints_np = torch.stack([torch.tensor(target['keypoints']).numpy() for target in targets]).reshape(-1, 16, 2)
            
            pck_score = calculate_pck(pred_keypoints_np, true_keypoints_np, threshold=5)  # using 5px as threshold
            total_pck += pck_score
            total_samples += 1

            # Backward pass and optimization
            #loss.backward()  # Backpropagate the loss
            loss = sum(loss for loss in loss_dict.values())  # Total loss (classification + bbox + keypoints)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model weights

            running_loss += loss.item()  # Accumulate loss for averaging

            # running_loss += loss.item()  # Accumulate loss for averaging
            
            # Log batch loss and PCK to TensorBoard
            writer.add_scalar("Batch/Loss", loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Batch/PCK", pck_score, epoch * len(train_loader) + batch_idx)
        
        # Compute epoch loss and log it
        epoch_loss = running_loss / len(train_loader)
        avg_pck = total_pck / total_samples
        writer.add_scalar("Epoch/Loss", epoch_loss, epoch)
        writer.add_scalar("Epoch/PCK", avg_pck, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, PCK: {avg_pck}%")

        # Save model checkpoint at the end of the epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

        # Visualize the predictions and ground truth keypoints on a sample image
        if epoch % 5 == 0:  # Every 5 epochs, log a sample visualization
            sample_image = images[0].cpu().detach().numpy()  # Take first image in batch
            predicted_keypoints = pred_keypoints[0].cpu().detach().numpy().reshape(16, 2)
            true_keypoints = targets[0]['keypoints']
            
            img_width, img_height = sample_image.shape[1], sample_image.shape[2]
            vis_image = visualize_keypoints(sample_image, predicted_keypoints, true_keypoints, img_width, img_height)
            writer.add_image('Keypoints/Visualization', vis_image, epoch)

    writer.close()