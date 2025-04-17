import torch                         # Main PyTorch Library
import torch.nn as nn                # Base class for all neural network modules.
import torch.nn.functional as F      # Functionan components for neural networks.
from torch.utils.data import Dataset # Creating and loading custom datasets.
import pandas as pd                  # For reading/parsing CSV files (image annotations).
from PIL import Image                # Reading/loading images.
import parameters as param

# Define a custom dataset
# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Initialize the custom dataset.

        Args:
            csv_file (str): Path to the CSV file containing image annotations.
            img_dir (str): Directory path where images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)    # Read annotations CSV file
        self.img_dir = img_dir                      # Image directory path
        self.transform = transform                  # Image transformer (for manipulating image sizes)

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset (size of annotations file).
        """
        return len(self.annotations)  # Size of annotations file

    def __getitem__(self, index):
        """
        Retrieve a sample from the dataset at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[image, torch.Tensor]: A tuple of the transformed image and its corresponding frequency annotations.
        """
        img_name = self.annotations.iloc[index, 0]  # Get the name of the image at the specified index
        image = Image.open(f"{self.img_dir}/{img_name}").convert('RGB')  # Open the image and convert to RGB if not already
        if self.transform:
            image = self.transform(image)  # Transform image (resize and change to Tensor type)
        frequencies = self.annotations.iloc[index, 1:].values.astype(float)  # Extract annotations (frequency response values)
        return image, torch.tensor(frequencies, dtype=torch.float32)  # Return the image and its annotation (both as tensors)

# Define Custom Neural Network
class CircuitFrequencyResponseModel(nn.Module):
    def __init__(self, output_length):
        """
        Initialize the Circuit Frequency Response Model.

        Args:
            output_length (int): Length of the model output.
        """
        super(CircuitFrequencyResponseModel, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(param.input_layers, param.Conv1_output_layers, kernel_size=param.conv_filter_size, padding=param.padding)  # First convolutional layer
        self.conv2 = nn.Conv2d(param.Conv1_output_layers, param.Conv2_output_layers, kernel_size=param.conv_filter_size, padding=param.padding)  # Second convolutional layer
        self.conv3 = nn.Conv2d(param.Conv2_output_layers, param.Conv3_output_layers, kernel_size=param.conv_filter_size, padding=param.padding)  # Third convolutional layer
        
        self.pool = nn.MaxPool2d(kernel_size=param.maxp_filter_size, stride=param.maxp_stride)  # Max pooling layer
        
        # Fully connected layers
        self.fc1 = nn.Linear(param.conv_flat_output, param.Lin1_output_layers)  # First fully connected layer
        self.fc2 = nn.Linear(param.Lin1_output_layers, param.Lin2_output_layers)  # Second fully connected layer
        self.fc3 = nn.Linear(param.Lin2_output_layers, param.Lin3_output_layers)  # Third fully connected layer
        self.fc4 = nn.Linear(param.Lin3_output_layers, param.Lin4_output_layers)  # Fourth fully connected layer
        self.fc5 = nn.Linear(param.Lin4_output_layers, param.Lin5_output_layers)  # Fifth fully connected layer
        
        self.dropout = nn.Dropout(p=param.dropout_prob)  # Dropout layer to prevent overfitting
        
        # Output layer
        # Originally commented out, using Lin3_output_layers
        # self.fc_out = nn.Linear(Lin3_output_layers, output_length)
        self.fc_out = nn.Linear(param.Lin5_output_layers, output_length)  # Linear layer for the final output
        
        self.output_length = output_length  # Record the output length for reference

    def forward(self, x):
        """
        Define the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor representing a batch of images.
        
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        # Forward pass through convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten feature map before fully connected layers
        x = x.view(x.size(0), -1)
        
        # Forward pass through fully connected layers with ReLU and dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        
        # Final output layer
        x = self.fc_out(x)
        
        return x

if __name__ == "__main__":
    print("This file is useless when run standalone!")