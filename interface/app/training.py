import torch                                            # Main PyTorch Library
import torch.nn as nn                                   # Base class for all neural network modules.
from torch.utils.data import DataLoader                 # Creating and loading custom datasets.
from torch.amp import GradScaler, autocast              # Used for optimizing training on CUDA enabled devices.
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Used for learning rate optimization
import os                                               # Used for modifying environment variables.
import time                                             # For testing final NN.
import neural_network as cap_nn
import parameters as param

class LRResetScheduler:
    def __init__(self, optimizer, initial_lr, reset_interval, lr_factor, decay_factor, lr_threshold):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.reset_interval = reset_interval
        self.lr_factor = lr_factor
        self.decay_factor = decay_factor
        self.lr_threshold = lr_threshold
        self.epoch = 0

    def step(self):
        if self.epoch % self.reset_interval == 0:
            new_lr = self.initial_lr + self.initial_lr * (self.lr_factor ** (self.epoch // self.reset_interval))
            if new_lr > self.lr_threshold:
                new_lr = self.initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Epoch {self.epoch}: Learning rate reset to {new_lr:.4f}")
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.decay_factor
        self.epoch += 1

# Training loop
def training_loop(csv_file: str, img_dir: str, model_name: str) -> float:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Optimizing memory usage.
    dataset = cap_nn.CapstoneDataset(csv_file=csv_file, img_dir=img_dir, transform=param.transform)
    train_loader = DataLoader(dataset, batch_size=param.training_batch_size, shuffle=True)
    
    # Model parameter instantiation
    model = cap_nn.Capstone_CNN(num_outputs=param.num_outputs)                  # Instantiate the NN model.
    criterion = nn.MSELoss()                                                    # Instantiate the loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)                  # Instantiate the optimizer function.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)# Learning rate scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # Check what device to use (CUDA accelerator or CPU).
    scaler = GradScaler(device)                                                 # Mixed precision scaler
    
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
        scheduler.step(running_loss / len(train_loader))
        print(f'Epoch [{epoch+1}/{param.num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Save the model
    torch.save(model.state_dict(), f'{model_name}.pth')

    # Used for debugging/refactoring process (track total training time).
    end_time = time.time()
    return round(end_time - start_time, 2)

if __name__ == "__main__":
    print("This file is useless when run standalone!")