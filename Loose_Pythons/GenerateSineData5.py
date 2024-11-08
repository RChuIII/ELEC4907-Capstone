import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from concurrent.futures import ProcessPoolExecutor
import time

# Create directory to save images
if not os.path.exists('./images'):
    os.makedirs('./images')

def generate_sine_wave_data():
    # Randomly choose values for a, b, h, and k
    a = round(random.uniform(0.5, 3.0), 2)   # amplitude
    b = round(random.uniform(0.5, 2.0), 2)   # scaling factor for x-axis (frequency)
    h = round(random.uniform(-np.pi, np.pi), 2)  # horizontal shift
    k = round(random.uniform(-2, 2), 2)      # vertical shift

    # Define the sine wave function
    x = np.linspace(0, 2 * np.pi, 400)  # Generate x values from 0 to 2*pi
    y = a * np.sin((x - h) / b) + k       # Apply the sine wave equation

    # Plot the sine wave with smaller size
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot(x, y, label=f'y={a}*sin((x-{h})/{b})+{k}', color='b')

    # Remove axes and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)

    plt.xlim(0, 2 * np.pi)
    plt.ylim(-4, 4)

    # Image file name format
    image_name = f"sin_{a}_{h}_{b}_{k}.png"
    image_path = os.path.join('./images', image_name)

    plt.savefig(image_path, dpi=150, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

    # Collect sine wave data (40 evenly spaced points between 0 and 2*pi)
    x_vals = np.linspace(0, 2 * np.pi, 40)
    y_vals = a * np.sin((x_vals - h) / b) + k

    # Round all values to 4 decimal places
    x_vals = np.round(x_vals, 4)
    y_vals = np.round(y_vals, 4)

    return image_name, list(y_vals)

def main(num_iters: int):
    # Create a CSV file to store the data
    csv_file = './annotations.csv'
    csv_columns = ['Image Name'] + [f'Point_{i+1}' for i in range(40)]

    # Initialize an empty list to store all the rows
    data = []

    # Generate 100 unique sine waves
    generated_sets = set()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(generate_sine_wave_data): i for i in range(num_iters)}

        for future in futures:
            image_name, wave_data = future.result()

            # Ensure no repeat of randomized parameters
            if image_name not in generated_sets:
                generated_sets.add(image_name)
                data.append([image_name] + wave_data)

    # Save the data to CSV
    df = pd.DataFrame(data, columns=csv_columns)
    df = df.round(4)
    df.to_csv(csv_file, index=False)

    print(f"Data saved to {csv_file} with {len(data)} entries.")

if __name__ == "__main__":
    start_time = time.time()  # Record start time
    main(500)
    end_time = time.time()    # Record end time

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
