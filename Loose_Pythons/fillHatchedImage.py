import cv2
# import numpy as np

def fill_hatched_shape(image_path, fill_color=(0, 0, 255)):
    """
    Fills a hatched shape in an image with a solid color.

    Parameters:
    - image_path: Path to the input image with the hatched shape.
    - fill_color: A tuple (B, G, R) defining the color to fill the shape with. Default is red (0, 0, 255).
    
    Returns:
    - The modified image with the hatched shape filled.
    """

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image (0 for background, 255 for the shape)
    _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the hatched shape
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contours found.")
        return image

    # Fill the contours with the specified color
    for contour in contours:
        cv2.drawContours(image, [contour], -1, fill_color, thickness=cv2.FILLED)

    return image

def save_image(image, output_path):
    """Saves the modified image to the specified output path."""
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    input_image_path = './input.png'  # Replace with your image file
    output_image_path = './output.png'  # Where to save the output image

    # Fill the hatched shape with a solid color (red in this case)
    filled_image = fill_hatched_shape(input_image_path, fill_color=(0, 0, 255))

    if filled_image is not None:
        save_image(filled_image, output_image_path)
