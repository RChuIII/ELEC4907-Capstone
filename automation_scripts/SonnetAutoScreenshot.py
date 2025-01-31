import pyautogui
import numpy as np
import time
from itertools import product
from PIL import Image

class Circuits:
    settings_button_location = [1020,130]
    initial_variable_location = [1020,500]
    variable_button_distance = 29
    off_board_location = [250,300]
    board_locations = [[402,312] , [1945,1343]]
    ok_settings_button_location = [1500,920]

    LPF_params_small = [
        ["length", (100, 140, 10)],
        ["width", (10, 40, 7.5)],
        ["height", (40, 80, 10)],
        ["sep", (10, 50, 10)]
    ]

    HPF_params = [
        ["width", (10, 40, 7.5)],
        ["width2", (10, 40, 7.5)],
        ["height", (50, 150, 25)],
        ["sep", (40, 70, 7.5)]
    ]
        
    BPF_params = [
        ["length", (170, 230, 15)],
        ["length2", (120, 160, 10)],
        ["height", (20, 40, 5)],
        ["sep", (20, 80, 15)]
    ]
    Notch_params_small = [
        ["length", (100, 200, 25)],
        ["width", (20, 40, 5)],
        ["height", (20, 40, 5)],
        ["sep", (80, 120, 10)]
    ]


def generate_sweep_parameters(parameters: list) -> list:
    ranges = []
    for parameter in parameters:
        param_min, param_max, step_size = parameter[1]
        ranges.append(list(map(int, np.arange(param_min, param_max + 1, step_size))))
    
    all_params = list(product(*ranges))
    return all_params

def click(location: list, timing: float) -> None:
    x, y = location         # XY co-ords  
    pyautogui.click(x, y)   # 👆
    time.sleep(timing / 2)         # 💤
    return None

def double_click(location: list, timing: float, x_diff=0,y_diff=0) -> None:
    x, y = location         # XY co-ords  
    pyautogui.click(x+x_diff, y+y_diff)   # 👆
    time.sleep(0.01)        # 💤
    pyautogui.click(x+x_diff, y+y_diff)   # 👆
    time.sleep(timing / 2)         # 💤
    return None

def take_pic(name: str, bottom_right_corner: list, top_left_corner: list, timing: float, bit_depth=4) -> None:
    screenshot = pyautogui.screenshot()                     # Take initial screenshot
    screenshot = screenshot.convert("RGB")                  # Convert to RGB colour space
    x1, y1 = bottom_right_corner                            # 
    x2, y2 = top_left_corner                                # 
    cropped_image = screenshot.crop((x1, y1, x2, y2))       # Crop the image from the tlc and brc
    cropped_image = cropped_image.convert("P", palette=Image.ADAPTIVE, colors=(2**bit_depth))   # Convert image to lower colour depth
    cropped_image.save(f"./{name}.png")                     # Save image
    time.sleep(timing/2)                                         # 💤
    return None

def grab_screenshot(circuit_name: str, 
                    parameters: list, 
                    parameter_names: list, 
                    loc_settings: list, 
                    loc_variable: list, 
                    dif_variable: int, 
                    loc_offBoard: list, 
                    loc_boardCorners: list,
                    loc_okCloseSettings: list,
                    timing: float
                    ) -> None:
    out_name = ''
    click(loc_settings, timing=5)                                             # Open settings window
    for i in range(len(parameters)):                            # For each parameter...
        param_name = parameter_names[i]                             # Get parameter name
        out_name = out_name + f'{param_name}{parameters[i]}_'       # Append to image name
        double_click(loc_variable, timing, y_diff=(dif_variable * i))             # Click Variable from variable list
        pyautogui.press('tab')                                      # Move cursor to variable value text box
        time.sleep(timing/2)                                             # 💤
        pyautogui.write(str(parameters[i]))                         # Write the variable value in the text box
        time.sleep(timing/2)                                             # 💤
        pyautogui.press('enter')                                    # Close variable window
        time.sleep(timing/2)                                             # 💤
    click(loc_okCloseSettings, timing)                                      # Click OK (i.e. exit settings window)
    click(loc_offBoard, timing)                                             # Move cursor out of the screenshot area
    take_pic(f'./automation_scripts/Images/{circuit_name}/{circuit_name}_{out_name}'[:-1],     # Take and save image
            loc_boardCorners[0], 
            loc_boardCorners[1],
            timing)
    return None

def main(circuit_name: str, circuit_params: list) -> None:
    combinations = generate_sweep_parameters(circuit_params)    # Create all possible circuit variations
    parameter_names = [param[0] for param in circuit_params]    # Get parameter names 

    for combination in combinations:                            # For each variation, take a screenshot of the circuit
        # print(combination)
        grab_screenshot(circuit_name=circuit_name,
                        parameters=combination,
                        parameter_names=parameter_names,
                        loc_settings=Circuits.settings_button_location,
                        loc_variable=Circuits.initial_variable_location,
                        dif_variable=Circuits.variable_button_distance,
                        loc_offBoard=Circuits.off_board_location,
                        loc_boardCorners=Circuits.board_locations,
                        loc_okCloseSettings=Circuits.ok_settings_button_location,
                        timing=1
        )
    return None

if __name__ == "__main__":
    time.sleep(5)                                   # 💤
    main("LowPassFilter", Circuits.LPF_params_small)      # Get screenshots for LPF
    # time.sleep(5)                                   # 💤
    # main("HighPassFilter", Circuits.HPF_params)     # Get screenshots for HPF
    # time.sleep(5)                                   # 💤
    # main("BandPassFilter", Circuits.BPF_params)     # Get screenshots for BPF
    # time.sleep(5)                                   # 💤
    # main("NotchFilter", Circuits.Notch_params_half)      # Get screenshots for NF
