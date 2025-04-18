import torch                                            # Main PyTorch Library
import torch.nn as nn                                   # Base class for all neural network modules.
from torch.utils.data import DataLoader                 # Creating and loading custom datasets.
from torch.amp import GradScaler, autocast              # Used for optimizing training on CUDA enabled devices.
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Used for learning rate optimization
import os                                               # Used for modifying environment variables.
import time                                             # For testing final NN.
from tqdm import tqdm
import neural_network as cap_nn
import parameters as param

# Training loop
def training_loop(train_csv_file: str, train_img_dir: str, val_csv_file: str, val_img_dir: str, model_name: str) -> float:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Optimizing memory usage.
    # Create and load dataset
    train_dataset = cap_nn.CustomDataset(csv_file=train_csv_file, img_dir=train_img_dir, transform=param.transform)  # Initialize training dataset with transformations
    val_dataset = cap_nn.CustomDataset(csv_file=val_csv_file, img_dir=val_img_dir, transform=param.transform)        # Initialize validation dataset

    # Create data loaders for training and validation datasets
    train_loader = DataLoader(
        train_dataset,  # Dataset for training
        batch_size=16,  # Number of samples per batch for training
        shuffle=True    # Shuffle training data after every epoch
    )
    val_loader = DataLoader(
        val_dataset,    # Dataset for validation
        batch_size=4,   # Number of samples per batch for validation
        shuffle=True    # Shuffle validation data to ensure randomness
    )
    
    # Training Variables (Define the model, loss function, optimizer, and scheduler)
    model = cap_nn.CircuitFrequencyResponseModel(output_length=param.num_outputs)  # Initialize the model with the specified output length
    criterion = nn.MSELoss()  # Define the loss function (Mean Squared Error Loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)  # Initialize the optimizer (Adam) with a learning rate; alternatives commented
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( # Initialize a learning rate scheduler to reduce the learning rate when a plateau in validation loss is observed.
        optimizer,            # Optimizer whose learning rate will be adjusted
        mode='min',           # Mode of optimization ('min' since lower validation loss is better)
        factor=0.75,          # Factor by which the learning rate will be reduced
        patience=20,          # Number of epochs with no improvement after which learning rate will be reduced
        threshold=0.001,      # Threshold for measuring the new optimum for early stopping
        verbose=True          # Print a message when learning rate is updated
    )
    # Move model to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if CUDA is available, use GPU if so, else use CPU
    model.to(device)  # Transfer model to the selected device

    start_epoch = 0  # Starting epoch number, useful for resuming training
    num_epochs = 600  # Total number of epochs for training
        

    # Training loop
    start_time = time.time()                                            # Used for debugging/refactoring process (track total training time).
    # Loop over the dataset multiple times, where each loop is a complete pass over the data (an epoch)
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode to ensure training-specific operations are active
        running_loss = 0.0  # Initialize the running loss to accumulate total loss for the epoch
        
        # Iterate over batches of images and targets from the training data
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, targets = images.to(device), targets.to(device)  # Move images and targets to the designated device (CPU or GPU)
            optimizer.zero_grad()     # Clear the gradients of all optimized tensors
            
            outputs = model(images)   # Forward pass: compute the model output for the batch of images
            loss = criterion(outputs.squeeze(-1), targets)  # Compute the loss between model predictions and actual targets
            
            loss.backward()  # Backward pass: compute gradients of the loss with respect to model parameters
            optimizer.step()  # Update model parameters based on the computed gradients
            
            running_loss += loss.item()  # Accumulate the batch loss into running loss
        
        # Compute the average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")  # Print the average training loss for the epoch
        
        model.eval()  # Set the model to evaluation mode, affecting certain layers which behave differently during evaluation (e.g., dropout)
        val_loss = 0.0  # Initialize validation loss accumulator
        
        # Disable gradient computation for validation to save memory and computation
        with torch.no_grad():
            # Iterate over validation dataset
            for val_images, val_targets in val_loader:
                val_images, val_targets = val_images.to(device), val_targets.to(device)  # Move validation data to the device
                val_outputs = model(val_images)  # Compute model output for validation data
                v_loss = criterion(val_outputs.squeeze(-1), val_targets)  # Calculate validation loss
                val_loss += v_loss.item()  # Accumulate validation loss
        
        # Compute the average validation loss for the epoch
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")  # Print the average validation loss
        
        scheduler.step(avg_val_loss)  # Update the learning rate based on validation loss
        
        # Save a checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'epoch': epoch + 1,                        # Current epoch number
                'model_state_dict': model.state_dict(),    # Model parameters
                'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
                'scheduler_state_dict': scheduler.state_dict(),  # Scheduler state
                'loss': avg_loss                           # Latest training loss
            }
            torch.save(checkpoint, f"Checkpoint_epoch_{epoch + 1}.pth")  # Save the checkpoint to a file
            print(f"Checkpoint saved at epoch {epoch + 1}")  # Notify that checkpoint is saved

    print("Training complete!")  # Indicate that training has finished

    # Save the final model parameters to a file
    torch.save(model.state_dict(), f'CapstoneFull_Complete_V3.pth')

    # Used for debugging/refactoring process (track total training time).
    end_time = time.time()
    return round(end_time - start_time, 2)

if __name__ == "__main__":
    print("This file is useless when run standalone!")