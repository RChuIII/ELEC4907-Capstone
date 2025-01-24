import pyautogui
import numpy as np
import time
from itertools import product
from PIL import Image

class Circuits:
    settings_button_location = [0,0]
    initial_variable_location = [0,0]
    variable_button_distance = 0
    off_board_location = [0,0]
    board_locations = [[0,0] , [0,0]]
    ok_settings_button_location = [0,0]

    LPF_params = [
        ["length", (10, 37, 3)],    # 40 - 10 = 30 / 3 = 10 points ✅
        ["width", (100, 136, 4)],   # 140 - 100 = 40 / 4 = 10 points ✅
        ["height", (40, 76, 4)],    # 80 - 40 = 40 / 4 = 10 points ✅
        ["sep", (10, 46, 4)]        # 50 - 10 = 40 / 4 = 10 points ✅
    ]

    HPF_params = [
        ["width", (10, 37, 3)],     # 40 - 10 = 30 / 3 = 10 points ✅
        ["width2", (10, 37, 3)],    # 40 - 10 = 30 / 3 = 10 points ✅
        ["height", (50, 140, 10)],  # 150 - 50 = 100 / 10 = 10 points ✅
        ["sep", (40, 67, 3)]        # 70 - 40 = 30 / 3 = 10 points ✅
    ]
        
    BPF_params = [
        ["length", (170, 226, 6)],  # 230 - 170 = 60 / 6 = 10 points ✅
        ["length2", (120, 156, 4)], # 160 - 120 = 40 / 4 = 10 points ✅
        ["height", (20, 38, 2)],    # 40 - 20 = 20 / 2 = 10 points ✅
        ["sep", (20, 74, 6)]        # 80 - 20 = 60 / 6 = 10 points  ✅
    ]

    Notch_params = [
        ["length", (100, 190, 10)], # 200 - 100 = 100 / 10 = 10 points ✅
        ["width", (20, 38, 2)],     # 40 - 20 = 20 / 2 = 10 points ✅
        ["height", (20, 38, 2)],    # 40 - 20 = 20 / 2 = 10 points ✅
        ["sep", (80, 116, 4)]       # 120 - 80 = 40 / 4 = 10 points ✅
    ]


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
                        loc_settings=Circuits.settings_button_location,
                        loc_variable=Circuits.initial_variable_location,
                        dif_variable=Circuits.variable_button_distance,
                        loc_offBoard=Circuits.off_board_location,
                        loc_boardCorners=Circuits.board_locations,
                        loc_okCloseSettings=Circuits.ok_settings_button_location
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