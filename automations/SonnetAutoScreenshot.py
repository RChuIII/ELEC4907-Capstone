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
    time.sleep(0.5)
    return None

def double_click(location: list) -> None:
    x, y = location
    pyautogui.click(x, y)
    time.sleep(0.01)
    pyautogui.click(x, y)
    time.sleep(0.5)
    return None

def take_pic(name: str, top_right_corner: list, top_left_corner: list , bit_depth=4) -> None:
    screenshot = pyautogui.screenshot()
    screenshot = screenshot.convert("RGB")
    x1, y1 = top_right_corner
    x2, y2 = top_left_corner
    cropped_image = screenshot.crop((x1, y1, x2, y2))
    cropped_image = cropped_image.convert("P", palette=Image.ADAPTIVE, colors=(2**bit_depth))
    cropped_image.save(f"./{name}.png")
    time.sleep(0.5)
    return None

def grab_screenshot(circuit_name: str, 
                    parameters: list, 
                    parameter_names: list, 
                    loc_settings: list, 
                    loc_variable: list, 
                    dif_variable: int, 
                    loc_offBoard: list, 
                    loc_boardCorners: list,
                    loc_okCloseSettings: list
                    ) -> None:
    out_name = ''
    click(loc_settings)                                             # Click settings button (click xy)
    for i in range(len(parameters) - 1):                            # For each parameter...
        param_name = parameter_names[i]                             # Get parameter name
        out_name = out_name + f'{param_name}{parameters[i]}_'       # Append to image name
        double_click(loc_variable + (dif_variable * i))             # Click Variable from variable list
        #something about editing the variable
    click(loc_okCloseSettings)                                      # Click OK (i.e. exit settings window)
    click(loc_offBoard)
    take_pic(f'./{circuit_name}/{circuit_name}_{out_name}'[-1], loc_boardCorners[0], loc_boardCorners[1])
    return None



def main(circuit_name: str, circuit_params: list) -> None:
    combinations = generate_sweep_parameters(circuit_params)
    
    for combination in combinations:
        grab_screenshot()
    
    return None

if __name__ == "__main__":
    time.sleep(5)
    main()