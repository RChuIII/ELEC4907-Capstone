import torchvision.transforms as transforms         # Transformers for image data.

def calculate_linears(num_layers: int, dim: list, conv_filter_size: int, conv_stride: int, maxp_filter_size: int, maxp_stride: int, padding: int) -> list:
    """
    Calculate the output dimensions of a series of convolutional and max pooling layers
    (Calculates the number of input neurons for the linear fully-connected layer).

    Args:
        num_layers (int): Number of convolutional and pooling layers.
        dim (list): Initial dimensions [width, height] of the input.
        conv_filter_size (int): Size of the convolutional filter (assumed square).
        conv_stride (int): Stride for the convolutional layers.
        maxp_filter_size (int): Size of the max pooling filter (assumed square).
        maxp_stride (int): Stride for the max pooling operation.
        padding (int): Amount of zero-padding added to both sides of the input.

    Returns:
        list: Final dimensions [width, height] after all layers are applied.
    """
    output = [dim[0], dim[1]]  # Initialize with the input dimensions (width, height)
    for i in range(num_layers):
        # Calculate convolution output dimensions using the given formula
        conv_output_width = ((output[0] - conv_filter_size + (2 * padding)) / conv_stride) + 1
        conv_output_height = ((output[1] - conv_filter_size + (2 * padding)) / conv_stride) + 1
        
        # Update output to convolution result dimensions
        output = [conv_output_width, conv_output_height]
        
        # Calculate max pooling output dimensions using the given formula
        pool_output_width = ((output[0] - maxp_filter_size) / maxp_stride) + 1
        pool_output_height = ((output[1] - maxp_filter_size) / maxp_stride) + 1
        
        # Update output to pooling result dimensions, converting to int for layer compatibility
        output = [int(pool_output_width), int(pool_output_height)]
    
    return output  # Return final dimensions after all layers are applied


image_width = 450                       # Set image width to be used for resizing images
image_height = 300                      # Set image height to be used for resizing images
train_img_dir = './images'              # Path to directory containing images for training
train_csv_file = './annotations.csv'    # Path to CSV file containing annotations for the images
val_img_dir = './val_images'            # Path to directory containing images for training
val_csv_file = './val_annotations.csv'  # Path to CSV file containing annotations for the images
num_outputs = 381                       # Number of output classes or target values for the model
batch_size = 16

# Define image transformer (changes input image dimensions and changes them to tensors)
transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),  # Resize images to the specified height and width
    transforms.ToTensor(),                           # Convert images to PyTorch tensors
])

# Circuit Convolution neural network configuration
input_layers = 3  # Number of input layers, typically corresponding to the color channels of an image (e.g., RGB)
Conv1_output_layers = 32  # Number of output channels for the first convolutional layer
Conv2_output_layers = Conv1_output_layers * 2  # Number of output channels for the second convolutional layer, doubling the previous layer
Conv3_output_layers = Conv2_output_layers * 2  # Number of output channels for the third convolutional layer, doubling again

conv_filter_size = 3  # Size of the convolutional filters (3x3)
conv_stride = 1  # Stride of the convolutional layer
maxp_filter_size = 2  # Size of the max pooling filters (2x2)
maxp_stride = 2  # Stride of the max pooling layer
padding = 1  # Padding added to the input of convolutional layers

# Calculate output dimensions after convolutional and pooling layers
linear_layers = calculate_linears(
    num_layers=3,  # Number of convolutional/pooling layers
    dim=[image_width, image_height],  # Initial image dimensions
    conv_filter_size=conv_filter_size,  # Convolution filter size
    conv_stride=conv_stride,  # Convolution stride
    maxp_filter_size=maxp_filter_size,  # Max pooling filter size
    maxp_stride=maxp_stride,  # Max pooling stride
    padding=padding  # Padding for convolutional layers
)

conv_flatten_widths = linear_layers[0]  # Final width after the convolutional layers
conv_flatten_heights = linear_layers[1]  # Final height after the convolutional layers

# Configuration for fully connected (linear) layers
Lin1_output_layers = 1024  # Number of output units for the first linear layer
Lin2_output_layers = Lin1_output_layers  # Number of output units for the second linear layer
Lin3_output_layers = Lin2_output_layers  # Number of output units for the third linear layer
Lin4_output_layers = Lin3_output_layers // 2  # Number of output units for the fourth linear layer, halving the previous layer
Lin5_output_layers = Lin4_output_layers // 2  # Number of output units for the fifth linear layer, halving again

dropout_prob = 0.1  # Probability for dropout, a regularization technique

# Calculate the number of neurons for the fully-connected layer after flattening
conv_flat_output = conv_flatten_widths * conv_flatten_heights * Conv3_output_layers  # Total number of neurons

if __name__ == "__main__":
    print("This file is useless when run standalone!")