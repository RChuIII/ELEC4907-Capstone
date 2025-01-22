import pyautogui
import numpy as np
import time
from itertools import product
from PIL import Image

class Circuits:
    LPF_params = []
    HPF_params = []
    BPF_params = []
    Notch_params = []

def generate_sweep_parameters(parameters: list) -> list:
    ranges = []
    for parameter in parameters:
        param_min, param_max, step_size = parameter[1]
        ranges.append(list(map(int, np.arange(param_min, param_max + 1, step_size))))
    
    all_params = list(product(*ranges))
    return all_params

def click(location: list) -> None:
    x, y = location         # XY co-ords  
    pyautogui.click(x, y)   # 👆
    time.sleep(0.5)         # 💤
    return None

def double_click(location: list) -> None:
    x, y = location         # XY co-ords  
    pyautogui.click(x, y)   # 👆
    time.sleep(0.01)        # 💤
    pyautogui.click(x, y)   # 👆
    time.sleep(0.5)         # 💤
    return None

def take_pic(name: str, bottom_right_corner: list, top_left_corner: list , bit_depth=4) -> None:
    screenshot = pyautogui.screenshot()                     # Take initial screenshot
    screenshot = screenshot.convert("RGB")                  # Convert to RGB colour space
    x1, y1 = bottom_right_corner                            # 
    x2, y2 = top_left_corner                                # 
    cropped_image = screenshot.crop((x1, y1, x2, y2))       # Crop the image from the tlc and brc
    cropped_image = cropped_image.convert("P", palette=Image.ADAPTIVE, colors=(2**bit_depth))   # Convert image to lower colour depth
    cropped_image.save(f"./{name}.png")                     # Save image
    time.sleep(0.5)                                         # 💤
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
    click(loc_settings)                                             # Open settings window
    for i in range(len(parameters) - 1):                            # For each parameter...
        param_name = parameter_names[i]                             # Get parameter name
        out_name = out_name + f'{param_name}{parameters[i]}_'       # Append to image name
        double_click(loc_variable + (dif_variable * i))             # Click Variable from variable list
        pyautogui.press('tab')                                      # Move cursor to variable value text box
        time.sleep(0.5)                                             # 💤
        pyautogui.write(str(parameters[i]))                         # Write the variable value in the text box
        time.sleep(0.5)                                             # 💤
        pyautogui.press('enter')                                    # Close variable window
        time.sleep(0.5)                                             # 💤
    click(loc_okCloseSettings)                                      # Click OK (i.e. exit settings window)
    click(loc_offBoard)                                             # Move cursor out of the screenshot area
    take_pic(f'./{circuit_name}/{circuit_name}_{out_name}'[-1],     # Take and save image
            loc_boardCorners[0], 
            loc_boardCorners[1])
    return None

def main(circuit_name: str, circuit_params: list) -> None:
    combinations = generate_sweep_parameters(circuit_params)    # Create all possible circuit variations
    parameter_names = [param[0] for param in circuit_params]    # Get parameter names 

    for combination in combinations:                            # For each variation, take a screenshot of the circuit
        grab_screenshot(circuit_name=circuit_name,
                        parameters=combination,
                        parameter_names=parameter_names,
                        loc_settings=[0,0],
                        loc_variable=[0,0],
                        dif_variable=0,
                        loc_offBoard=[0,0],
                        loc_boardCorners=[0,0],
                        loc_okCloseSettings=[0,0]
        )
    return None

if __name__ == "__main__":
    time.sleep(5)                                   # 💤
    main("LowPassFilter", Circuits.LPF_params)      # Get screenshots for LPF
    time.sleep(5)                                   # 💤
    main("HighPassFilter", Circuits.HPF_params)     # Get screenshots for HPF
    time.sleep(5)                                   # 💤
    main("BandPassFilter", Circuits.BPF_params)     # Get screenshots for BPF
    time.sleep(5)                                   # 💤
    main("NotchFilter", Circuits.Notch_params)      # Get screenshots for NF