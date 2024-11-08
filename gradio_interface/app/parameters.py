import torchvision.transforms as transforms         # Transformers for image data.

# Calculating number of input neurons for the linear fully-connected layer
def calculate_linears(num_layers: int, dim: list, conv_filter_size: int, conv_stride: int, maxp_filter_size: int, maxp_stride: int, padding: int) -> list:
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

# Define global variables
image_width = 300
image_height = 150
num_outputs = 40
training_batch_size = 256  # Number of training points per batch. 
num_epochs = 50            # Number of training epochs

# Circuit Convolution neural network configuration
input_layers = 3                                # Defining the number of input layers (in a coloured image there are 3 colour chanels, therefore 3 input layers).
Conv1_output_layers = 32                        # Arbitrary number of feature layers for the first convolution layer.
Conv2_output_layers = Conv1_output_layers * 2   # Arbitrary number of feature layers
Conv3_output_layers = Conv2_output_layers * 2   # Arbitrary number of feature layers
conv_filter_size = 7                            # Convolution kernel/filter size (i.e. NxN filter to perform convolutions)
conv_stride = 1                                 # Stride (how many spaces per convolution the kernel moves).
maxp_filter_size = 2                            # Kernel/filter size
maxp_stride = 2                                 # Same as conv_stride
padding = 1                                     # Number of pixels ignored on the boundary of the image.
# Calculating the number of required linear layers (the number of neurons in the FC layer is proportional to the total convolution output features).
linear_layers = calculate_linears(num_layers = 3,
                                    dim = [image_width, image_height],
                                    conv_filter_size = conv_filter_size,
                                    conv_stride = conv_stride,
                                    maxp_filter_size = maxp_filter_size,
                                    maxp_stride = maxp_stride,
                                    padding = padding
                                    )
conv_flatten_widths = linear_layers[0]          # Final image width after all convolution layers.
conv_flatten_heights = linear_layers[1]         # Final image height after all convolution layers.
Lin1_output_layers = 256                        # Arbitrary selection of output layers for the linear layer.
Lin2_output_layers = Lin1_output_layers // 2    # Arbitrary selection of output layers for the linear layer.
rnn_hidden_size = 128                           # Number of hidden neurons in the RNN layer (chosen arbitrarily).
rnn_num_layers = 2                              # Number of stacked RNN layers.

# Define the size of the image that will be used in training / testing
transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
])


if __name__ == "__main__":
    print("This file is useless when run standalone!")