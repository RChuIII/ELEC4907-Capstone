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
        ranges.append(list(map(float, np.arange(param_min, param_max + 1, step_size))))
    
    all_params = list(product(*ranges))
    return all_params

def click(location: list, timing: float) -> None:
    x, y = location         # XY co-ords  
    pyautogui.click(x, y)   # 👆
    time.sleep(timing / 2)  # 💤
    return None

def double_click(location: list, timing: float, x_diff=0,y_diff=0) -> None:
    x, y = location                         # XY co-ords  
    pyautogui.click(x+x_diff, y+y_diff)     # 👆
    time.sleep(0.01)                        # 💤
    pyautogui.click(x+x_diff, y+y_diff)     # 👆
    time.sleep(timing / 2)                  # 💤
    return None

def take_pic(name: str, bottom_right_corner: list, top_left_corner: list, timing: float, bit_depth=4) -> None:
    screenshot = pyautogui.screenshot()                     # Take initial screenshot
    screenshot = screenshot.convert("RGB")                  # Convert to RGB colour space
    x1, y1 = bottom_right_corner                            # 
    x2, y2 = top_left_corner                                # 
    cropped_image = screenshot.crop((x1, y1, x2, y2))       # Crop the image from the tlc and brc
    cropped_image = cropped_image.convert("P", palette=Image.ADAPTIVE, colors=(2**bit_depth))   # Convert image to lower colour depth
    cropped_image.save(f"./{name}.png")                     # Save image
    time.sleep(timing/2)                                    # 💤
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
                    timing: float,
                    previous_combo: tuple
                    ) -> None:
    out_name = ''
    click(loc_settings, timing=5)                                       # Open settings window
    for i in range(len(parameters)):                                    # For each parameter...
        param_name = parameter_names[i]                                 # Get parameter name
        out_name = out_name + f'{param_name}{parameters[i]}_'           # Append to image name
        if previous_combo[i] == parameters[i]:                          # If a parameter is the same as the previous parameter value...
            continue                                                    # Skip to save time.
        else:
            double_click(loc_variable, timing, y_diff=(dif_variable * i))   # Click Variable from variable list
            pyautogui.press('tab')                                          # Move cursor to variable value text box
            time.sleep(timing/2)                                            # 💤
            pyautogui.write(str(parameters[i]))                             # Write the variable value in the t ext box
            time.sleep(timing/2)                                            # 💤
            pyautogui.press('enter')                                        # Close variable window
            time.sleep(timing/2)                                            # 💤
    click(loc_okCloseSettings, timing)                                  # Click OK (i.e. exit settings window)
    click(loc_offBoard, timing)                                         # Move cursor out of the screenshot area
    take_pic(f'./automation_scripts/Images/{circuit_name}/{circuit_name}_{out_name}'[:-1],     # Take and save image
            loc_boardCorners[0], 
            loc_boardCorners[1],
            timing)
    return None

# Function to check if a specific tuple exists in the file
def check_tuple_in_file(file_path, target_tuple):
    try:
        # Open the file for reading
        with open(file_path, 'r') as file:
            # Iterate through each line in the file
            for line in file:
                # Convert each line into a tuple (assuming it's a valid tuple format)
                try:
                    current_tuple = eval(line.strip())  # Use eval to interpret the string as a tuple
                    if current_tuple == target_tuple:
                        # print(f"Found the tuple: {target_tuple}")
                        return True
                except:
                    continue  # If the line can't be evaluated as a tuple, skip it
        # print(f"Tuple {target_tuple} not found in the file.")
        return False
    except FileNotFoundError:
        print("The file was not found.")
        return False


def main(circuit_name: str, circuit_params: list) -> None:
    combinations = generate_sweep_parameters(circuit_params)    # Create all possible circuit variations
    parameter_names = [param[0] for param in circuit_params]    # Get parameter names 
    file_path = './automation_scripts/done_list.txt'
    previous_combo = (0,0,0,0)
    f = open(file_path, 'a')
    count = 0
    for combination in combinations:                            # For each variation, take a screenshot of the circuit
        if check_tuple_in_file(file_path, combination) is False:
            count += 1
            f.write(str(combination) + '\n')
            grab_screenshot(circuit_name=circuit_name,
                parameters=combination,
                parameter_names=parameter_names,
                loc_settings=Circuits.settings_button_location,
                loc_variable=Circuits.initial_variable_location,
                dif_variable=Circuits.variable_button_distance,
                loc_offBoard=Circuits.off_board_location,
                loc_boardCorners=Circuits.board_locations,
                loc_okCloseSettings=Circuits.ok_settings_button_location,
                timing=3,
                previous_combo=previous_combo
            )
            previous_combo = combination
        if count == 100:
            break
    f.close()
    return None

if __name__ == "__main__":
    # time.sleep(5)                               # 💤
    # main("LPF", Circuits.LPF_params_small)      # Get screenshots for LPF
    # time.sleep(5)                             # 💤
    # main("HPF", Circuits.HPF_params)          # Get screenshots for HPF
    # time.sleep(5)                             # 💤
    # main("BPF", Circuits.BPF_params)          # Get screenshots for BPF
    time.sleep(5)                             # 💤
    main("notch", Circuits.Notch_params_small) # Get screenshots for NF
    