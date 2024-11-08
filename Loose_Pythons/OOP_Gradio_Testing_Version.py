import OOP_Gradio_Training_Version as CapstoneNN
import torch            # Pytorch main library
from PIL import Image   # Reading/loading images.

class Run_Prediction():
    def __init__(self, model_path: str) -> None:
        self.net = CapstoneNN.CustomCNN(num_outputs=CapstoneNN.Capstone_CNN_RNN_Trainer.num_outputs) # Initialize the NN
        self.net.load_state_dict(torch.load(model_path, weights_only=True)) # Load only the weights from the NN

    def predict_output(model, image_path, transform=CapstoneNN.Capstone_CNN_RNN_Trainer.transform):
        model.eval()                                                # Run model evalutaion
        with torch.no_grad():                                       # No gradient
            image = Image.open(image_path).convert('RGB')           # Open image and convert to RGB.
            image = transform(image).unsqueeze(0)                   # Transform image to tensor
            output = model(image)                                   # Get model output
            if output is not None:                                  # Model evaluation doesn't fail...
                predicted_outputs = output.cpu().numpy().flatten()  # Convert outptut to normal array.
                return predicted_outputs                            # Return prediction
            else:
                return None                                         # If evalutaion failed, return None.