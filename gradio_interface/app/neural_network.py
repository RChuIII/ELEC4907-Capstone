import torch                         # Main PyTorch Library
import torch.nn as nn                # Base class for all neural network modules.
import torch.nn.functional as F      # Functionan components for neural networks.
from torch.utils.data import Dataset # Creating and loading custom datasets.
import pandas as pd                  # For reading/parsing CSV files (image annotations).
from PIL import Image                # Reading/loading images.
import parameters as param

# Define a custom dataset
class CapstoneDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
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
    def __init__(self, num_outputs):
        super(Capstone_CNN, self).__init__()
        
        # Initializing each layer of the convolution neural network.
        self.conv1 = nn.Conv2d(param.input_layers, param.Conv1_output_layers, kernel_size=param.conv_filter_size, padding=param.padding)
        self.conv2 = nn.Conv2d(param.Conv1_output_layers, param.Conv2_output_layers, kernel_size=param.conv_filter_size, padding=param.padding)
        self.conv3 = nn.Conv2d(param.Conv2_output_layers, param.Conv3_output_layers, kernel_size=param.conv_filter_size, padding=param.padding)
        self.pool = nn.MaxPool2d(kernel_size=param.maxp_filter_size, stride=param.maxp_stride)
        self.fc1 = nn.Linear(int(param.Conv3_output_layers * param.conv_flatten_widths * param.conv_flatten_heights), param.Lin1_output_layers)
        self.fc2 = nn.Linear(param.Lin1_output_layers, param.Lin2_output_layers)
        self.rnn = nn.RNN(input_size=param.Lin2_output_layers, hidden_size=param.rnn_hidden_size, num_layers=param.rnn_num_layers, batch_first=True)
        self.fc_rnn = nn.Linear(param.rnn_hidden_size, num_outputs)

    # Defining the forward pass of the neural network.
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Reshape for RNN: (batch_size, seq_len, input_size)
        x = x.unsqueeze(1)  # Add sequence dimension (seq_len = 1)

        rnn_out, _ = self.rnn(x)  # Get the RNN output
        rnn_out = rnn_out[:, -1, :]  # Take the output of the last time step
        x = self.fc_rnn(rnn_out)  # Output layer for regression
        return x

if __name__ == "__main__":
    print("This file is useless when run standalone!")