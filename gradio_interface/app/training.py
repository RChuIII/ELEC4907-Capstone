import torch                                        # Main PyTorch Library
import torch.nn as nn                               # Base class for all neural network modules.
from torch.utils.data import DataLoader             # Creating and loading custom datasets.
from torch.amp import GradScaler, autocast          # Used for optimizing training on CUDA enabled devices.
import os                                           # Used for modifying environment variables.
import time                                         # For testing final NN.
import neural_network as cap_nn
import parameters as param

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Optimizing memory usage.

# Training loop
def training_loop(csv_file: str, img_dir: str, model_name: str) -> float:
    dataset = cap_nn.CapstoneDataset(csv_file=csv_file, img_dir=img_dir, transform=param.transform)
    train_loader = DataLoader(dataset, batch_size=param.training_batch_size, shuffle=True)
    
    # Model parameter instantiation
    model = cap_nn.Capstone_CNN(num_outputs=param.num_outputs)                           # Instantiate the NN model.
    criterion = nn.MSELoss()                                                # Instantiate the loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)              # Instantiate the optimizer function.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Check what device to use (CUDA accelerator or CPU).
    scaler = GradScaler(device)                                             # Mixed precision scaler
    
    if device == 'cuda':
        torch.cuda.empty_cache()    # Empty the cache (in case there is something loaded from previous training).
    model.to(device)                # Move model to the target device (CUDA or CPU).

    # Training loop
    start_time = time.time()                                            # Used for debugging/refactoring process (track total training time).
    for epoch in range(param.num_epochs):                                     # Train the model for `num_epochs`.
        model.train()                                                   # Begin training...
        running_loss = 0.0                                              # Reset epoch's running loss.
        for images, targets in train_loader:                            # Load training dataset.
            images, targets = images.to(device), targets.to(device)     # Move Batch to target device

            optimizer.zero_grad()                                       # Zero the gradients
            with autocast(str(device)):                                 # Use mixed precision
                outputs = model(images)                                 # Forward pass
                loss = criterion(outputs, targets)                      # Compute loss

            scaler.scale(loss).backward()                               # Backward pass
            scaler.step(optimizer)                                      # Optimize weights
            scaler.update()                                             # Update scaler
            running_loss += loss.item()                                 # Increment running loss.

    # Save the model
    torch.save(model.state_dict(), f'{model_name}.pth')

    # Used for debugging/refactoring process (track total training time).
    end_time = time.time()
    return round(end_time - start_time, 2)

if __name__ == "__main__":
    print("This file is useless when run standalone!")