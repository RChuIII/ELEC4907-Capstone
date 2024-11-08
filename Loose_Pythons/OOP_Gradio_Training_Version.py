import torch                                        # Main PyTorch Library
import torch.nn as nn                               # Base class for all neural network modules.
import torch.nn.functional as F                     # Functionan components for neural networks.
import torchvision.transforms as transforms         # Transformers for image data.
from torch.utils.data import DataLoader, Dataset    # Creating and loading custom datasets.
from torch.amp import GradScaler, autocast          # Used for optimizing training on CUDA enabled devices.
import pandas as pd                                 # For reading/parsing CSV files (image annotations).
from PIL import Image                               # Reading/loading images.
import os                                           # Used for modifying environment variables.

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Optimizing memory usage.

# Define a custom dataset
class CapstoneDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, transform=None) -> None:
        self.annotations = pd.read_csv(csv_file)    # Read annotations CSV file
        self.img_dir = img_dir                      # Image directory Path
        self.transform = transform                  # Image transformer (for manipulating image sizes)

    def __len__(self):
        return len(self.annotations) # Size of annotations file

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]                          # Get the name of image at index
        image = Image.open(f"{self.img_dir}/{img_name}").convert('RGB')     # Open the image and convert to RGB if not already
        if self.transform:
            image = self.transform(image)                                   # Transform image (resize and change to Tensor type)

        frequencies = self.annotations.iloc[index, 1:].values.astype(float) # Extract annotations (frequency response values)
        return image, torch.tensor(frequencies, dtype=torch.float32)        # Return the image and it's annotation (Both as tensors)

# Define Custom Neural Network
class Capstone_CNN(nn.Module):
    def __init__(self, 
        input_layers: int,
        Conv1_output_layers: int,
        Conv2_output_layers: int,
        Conv3_output_layers: int,
        padding: int,
        conv_filter_size: int,
        maxp_filter_size: int,
        maxp_stride: int,
        conv_flatten_widths: int,
        conv_flatten_heights: int,
        Lin1_output_layers: int,
        Lin2_output_layers: int,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        num_outputs: int) -> None:
        super(Capstone_CNN, self).__init__()

        # Initializing each layer of the convolution neural network.
        self.conv1 = nn.Conv2d(input_layers, Conv1_output_layers, kernel_size=conv_filter_size, padding=padding)
        self.conv2 = nn.Conv2d(Conv1_output_layers, Conv2_output_layers, kernel_size=conv_filter_size, padding=padding)
        self.conv3 = nn.Conv2d(Conv2_output_layers, Conv3_output_layers, kernel_size=conv_filter_size, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=maxp_filter_size, stride=maxp_stride)
        self.fc1 = nn.Linear(int(Conv3_output_layers * conv_flatten_widths * conv_flatten_heights), Lin1_output_layers)
        self.fc2 = nn.Linear(Lin1_output_layers, Lin2_output_layers)
        self.rnn = nn.RNN(input_size=Lin2_output_layers, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, batch_first=True)
        self.fc_rnn = nn.Linear(rnn_hidden_size, num_outputs)


    # Defining the forward pass of the neural network.
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Reshape for RNN: (batch_size, seq_len, input_size)
        x = x.unsqueeze(1)              # Add sequence dimension (seq_len = 1)
        rnn_out, _ = self.rnn(x)        # Get the RNN output
        rnn_out = rnn_out[:, -1, :]     # Take the output of the last time step
        x = self.fc_rnn(rnn_out)        # Output layer for regression
        return x


class Capstone_CNN_RNN_Trainer():
    def __init__(self, img_dir: str, csv_file: str) -> None:
        # Define global variables
        self.image_width = 300               # Training image width
        self.image_height = 150              # Training image height
        # self.img_dir = './images'            # Image directory
        # self.csv_file = './annotations.csv'  # Image annotations
        self.num_outputs = 40                # Number of desired output neurons
        self.training_batch_size = 64        # Number of images to process at a time per ephoch

        # Define the size of the image that will be used in training / testing
        self.transform = transforms.Compose([
            transforms.Resize((self.image_height, self.image_width)), # Resize image
            transforms.ToTensor(),                          # Raster(pixel image) > Tensor (big 3D matrix)
        ])

        dataset = CapstoneDataset(csv_file=csv_file, img_dir=img_dir, transform=self.transform)  # Load te images and their annotations.  
        self.train_loader = DataLoader(dataset, batch_size=self.training_batch_size, shuffle=True)    # Create data loader instance.

        # Circuit Convolution neural network configuration
        self.input_layers = 3                                # Defining the number of input layers (in a coloured image there are 3 colour chanels, therefore 3 input layers).
        self.Conv1_output_layers = 32                        # Arbitrary number of feature layers for the first convolution layer.
        self.Conv2_output_layers = self.Conv1_output_layers * 2   # Arbitrary number of feature layers
        self.Conv3_output_layers = self.Conv2_output_layers * 2   # Arbitrary number of feature layers
        self.conv_filter_size = 7                            # Convolution kernel/filter size (i.e. NxN filter to perform convolutions)
        self.conv_stride = 1                                 # Stride (how many spaces per convolution the kernel moves).
        self.maxp_filter_size = 2                            # Kernel/filter size
        self.maxp_stride = 2                                 # Same as conv_stride
        self.padding = 1                                     # Number of pixels ignored on the boundary of the image.
        # Calculating the number of required linear layers (the number of neurons in the FC layer is proportional to the total convolution output features).
        linear_layers = self.calculate_linears(num_layers = 3,
                                            dim = [self.image_width, self.image_height],
                                            conv_filter_size = self.conv_filter_size,
                                            conv_stride = self.conv_stride,
                                            maxp_filter_size = self.maxp_filter_size,
                                            maxp_stride = self.maxp_stride,
                                            padding = self.padding
                                            )
        self.conv_flatten_widths = linear_layers[0]          # Final image width after all convolution layers.
        self.conv_flatten_heights = linear_layers[1]         # Final image height after all convolution layers.
        self.Lin1_output_layers = 256                        # Arbitrary selection of output layers for the linear layer.
        self.Lin2_output_layers = self.Lin1_output_layers // 2    # Arbitrary selection of output layers for the linear layer.
        self.rnn_hidden_size = 256                           # RNN hidden layers.
        self.rnn_num_layers = 2                              # Define the number of stacked RNN layers. 

        # Training parameter instantiation
        self.learning_rate = 0.001                                                   # Model learning rate.
        self.num_epochs = 50                                                         # Number of training epochs.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Check what device to use (CUDA accelerator or CPU).
        # Instantiate the NN model.
        self.model = Capstone_CNN(
                input_layers = self.input_layers,
                Conv1_output_layers = self.Conv1_output_layers,
                Conv2_output_layers = self.Conv2_output_layers,
                Conv3_output_layers = self.Conv3_output_layers,
                padding = self.padding,
                conv_filter_size = self.conv_filter_size,
                maxp_filter_size = self.maxp_filter_size,
                maxp_stride = self.maxp_stride,
                conv_flatten_widths = self.conv_flatten_widths,
                conv_flatten_heights = self.conv_flatten_heights,
                Lin1_output_layers = self.Lin1_output_layers,
                Lin2_output_layers = self.Lin2_output_layers,
                rnn_hidden_size = self.rnn_hidden_size,
                rnn_num_layers = self.rnn_num_layers,
                num_outputs = self.num_outputs)
        self.criterion = nn.MSELoss()                                                # Instantiate the loss function.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)              # Instantiate the optimizer function.
        self.scaler = GradScaler(self.device)                                             # Mixed precision scaler

    # Calculating number of input neurons for the linear fully-connected layer
    def calculate_linears(self, num_layers: int, dim: list, conv_filter_size: int, conv_stride: int, maxp_filter_size: int, maxp_stride: int, padding: int) -> list:
        output = [dim[0], dim[1]]  # Initialize with the input dimensions

        for i in range(num_layers):
            # Calculate convolution output dimensions
            conv_output_width = ((output[0] - conv_filter_size + (2 * padding)) / conv_stride) + 1
            conv_output_height = ((output[1] - conv_filter_size + (2 * padding)) / conv_stride) + 1

            # Update output to convolution result
            output = [conv_output_width, conv_output_height]

            # Calculate max pooling output dimensions
            pool_output_width = ((output[0] - maxp_filter_size) / maxp_stride) + 1
            pool_output_height = ((output[1] - maxp_filter_size) / maxp_stride) + 1

            # Update output to pooling result
            output = [int(pool_output_width), int(pool_output_height)]

        return output  # Return final dimensions

    def Training_Loop(self, model_name: str) -> None:
        # Training loop
        if self.device == 'cuda':
            torch.cuda.empty_cache()    # Empty the cache (in case there is something loaded from previous training).
        self.model.to(self.device)                # Move model to the target device (CUDA or CPU).

        for epoch in range(self.num_epochs):                                     # Train the model for `num_epochs`.
            self.model.train()                                                   # Begin training...
            running_loss = 0.0                                              # Reset epoch's running loss.
            for images, targets in self.train_loader:                            # Load training dataset.
                images, targets = images.to(self.device), targets.to(self.device)     # Move Batch to target device

                self.optimizer.zero_grad()                                       # Zero the gradients
                with autocast(self.device):                                      # Use mixed precision
                    outputs = self.model(images)                                 # Forward pass
                    loss = self.criterion(outputs, targets)                      # Compute loss

                self.scaler.scale(loss).backward()                               # Backward pass
                self.scaler.step(self.optimizer)                                      # Optimize weights
                self.scaler.update()                                             # Update scaler
                running_loss += loss.item()                                 # Increment running loss.
        torch.save(self.model.state_dict(), f'{model_name}.pth')  # Save the model
