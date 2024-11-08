import pyautogui
import numpy as np
import time
from itertools import product

def click(location: list) -> None:
    x, y = location
    pyautogui.click(x, y)
    return None

def take_pic(name: str, top_right_corner: list, top_left_corner: list ) -> None:
    screenshot = pyautogui.screenshot()
    screenshot = screenshot.convert("RGB")
    x1, y1 = top_right_corner
    x2, y2 = top_left_corner
    cropped_image = screenshot.crop((x1, y1, x2, y2))
    cropped_image.save(f"./{name}.png")
    return None

def grab_screenshot(off_board_location: list, param_locations: list, params: tuple, top_right_corner: list, top_left_corner: list) -> None:
    out_name = ''
    for i in range(len(param_locations)):
        out_name = out_name + f'{params[i]}_'
        # Double click parameter
        click(param_locations[i])
        time.sleep(0.01)
        click(param_locations[i])
        time.sleep(1)
        pyautogui.press('tab')
        time.sleep(1)
        pyautogui.write(str(params[i]))
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(1)
        click(off_board_location)
    time.sleep(1)
    take_pic(f'circuit_{out_name}', top_right_corner, top_left_corner)
    time.sleep(1)
    return None

# Function to generate all parameter combinations dynamically
def sweep_parameters(parameter_ranges):
    # Generate the ranges for each parameter based on input (min, max, step)
    ranges = []
    for param_range in parameter_ranges:
        param_min, param_max, step_size = param_range
        ranges.append(list(map(int, np.arange(param_min, param_max + 1, step_size))))
    
    # Generate all combinations of parameters using product
    # `itertools.product` gives the Cartesian product of the provided parameter ranges
    all_combinations = list(product(*ranges))
    
    return all_combinations

def main():
    off_board_location = [1,1]
    param1_location = [1, 668]
    param2_location = [2, 668]
    param3_location = [3, 668]
    param4_location = [4, 668]
    # 1687 x 842 ~ 2x1 -> transform to 400x200 or 800x400
    top_right_corner = [331,406]
    top_left_corner = [2018,1248]

    locations = [
        off_board_location,
        param1_location,
        param2_location,
        param3_location,
        param4_location
    ]
    # Define the range for each parameter as a list of tuples
    # Each tuple contains (min_value, max_value, step_size) for the parameter
    parameters = [
        (0, 2, 1),
        (0, 2, 1),
        (0, 2, 1), 
        (0, 2, 1)
    ]

    # Generate all possible parameter combinations
    combinations = sweep_parameters(parameters)

    # Print the combinations
    for combination in combinations:
        grab_screenshot(off_board_location=off_board_location,
                        param_locations=locations, 
                        params=combination, 
                        top_right_corner=top_right_corner,
                        top_left_corner=top_left_corner
                       )

if __name__ == "__main__":
    main()