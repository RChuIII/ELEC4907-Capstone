def calculate_linears(num_layers: int, dim: list, conv_filter: int, padding: int, conv_stride, pool_filter: int, pool_stride: int) -> list:
    output = [dim[0],dim[1]]
    for i in range(num_layers):
        conv_output_width = (output[0] - conv_filter + 2*padding) / conv_stride + 1
        conv_output_height = (output[1] - conv_filter + 2*padding) / conv_stride + 1
        pool_output_width = (conv_output_width + 2*padding - pool_filter) / pool_stride + 1
        pool_output_height = (conv_output_height + 2*padding - pool_filter) / pool_stride + 1
        output=[(pool_output_width), (pool_output_height)]
    return output


def calculate_size(num_layers: int, dim: list, conv_filter_size: int, conv_stride: int, maxp_filter_size: int, maxp_stride: int, padding: int) -> list:
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



input_layers = 3
image_width, image_height = 224, 224
conv_filter_size = 3
conv_stride = 1
maxp_filter_size = 2
maxp_stride = 2
padding = 0

linear_layers = calculate_linears(3, 
                                    [image_width, image_height],
                                    conv_filter_size,
                                    padding,
                                    conv_stride,
                                    maxp_filter_size, 
                                    maxp_stride
                                    )
print(linear_layers)
linear_layers2 = calculate_size( num_layers = 3, 
                                    dim = [image_width, image_height],
                                    conv_filter_size = conv_filter_size,
                                    conv_stride = conv_stride,
                                    maxp_filter_size = maxp_filter_size, 
                                    maxp_stride = maxp_stride,
                                    padding = padding
                                    )
print(linear_layers2)