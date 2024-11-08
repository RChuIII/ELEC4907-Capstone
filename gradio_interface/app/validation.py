import torch                                        # Main PyTorch Library
from PIL import Image                               # Reading/loading images.
import os                                           # Used for modifying environment variables.
import random                                       # For testing final NN.
import neural_network as capstoneNN
import parameters as param

# Define function to predict output
def predict_output(model_path, image_path, transform=param.transform):              # NN model path
    net = capstoneNN.Capstone_CNN(num_outputs=param.num_outputs)                    # Initialize the NN
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))   # Load on CPU
    
    net.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)

        output = net(image)

        if output is not None:
            predicted_outputs = output.cpu().numpy().flatten()
            return predicted_outputs
        else:
            return None


# Used for choosing a random image to test neural network
def choose_random_file(directory):
    # Get a list of files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:               # Check if there are any files in the directory
        return None             # Return nothing if empty.
    return random.choice(files) # Choose a random file


# # Test model with some number of test images.
# test_images = 2
# for i in range(test_images):
#     image_path = f'./images_500/{choose_random_file("./images_500")}'       # Path to a random input image
#     predicted_values = predict_output(net, image_path, train.transform)           # Get predicted values
#     print(f"Predicted output values for {image_path}: {predicted_values}")  # Print predicted values


if __name__ == "__main__":
    print("This file is useless when run standalone!")