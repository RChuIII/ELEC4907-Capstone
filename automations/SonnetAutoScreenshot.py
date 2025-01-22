import pyautogui
import numpy as np
import time
from itertools import product
from PIL import Image

def generate_sweep_parameters(parameters: list) -> list:
    ranges = []
    for param in parameters:
        param_min, param_max, step_size = param_range[1]
        ranges.append(list(map(int, np.arange(param_min, param_max + 1, step_size))))
    
    all_params = list(product(*ranges))
    return all_params

def click(location: list) -> None:
    x, y = location
    pyautogui.click(x, y)
    return None

def grab_screenshot() -> None:
    # Change circuit parameter
    # File > export image > return
    # Repeat
    return None



def main(circuit_params: list) -> None:
    combinations = generate_sweep_parameters(circuit_params)
    
    for combination in combinations:
        grab_screenshot()
    
    return None

if __name__ == "__main__":
    time.sleep(5)
    main()