import csv
import re

def extract_values_from_filename(filepath, parameter_names, param_order):
    """
    Extract filter parameters (width, length, height) and construct the image filename.
    Example input: "/home/vrajshah4/Desktop/4907_Test/folded_double_stub_notch_filter_param.son DE_EMBEDDED width=20.0 length=190.0 height=22.0"
    Example output: ("folded_double_stub_notch_filter_param", "20", "190", "22")
    """
    # Extract width, length, and height using regex from the file path
    find_str = rf'{parameter_names[param_order[0] - 1]}=([\d.]+)\s+{parameter_names[param_order[1] - 1]}=([\d.]+)\s+{parameter_names[param_order[2] - 1]}=([\d.]+)\s+{parameter_names[param_order[3] - 1]}=([\d.]+)'
    match = re.search(find_str, filepath)

    if match:
        base_filename = filepath.split('/')[-1].split(' ')[0].replace('.son', '')
        formatted_params = {}
        for i in range(len(match.groups())):
            formatted_params[parameter_names[param_order[i] - 1]] = match.group(i+1)
        return base_filename, formatted_params
    return None, None

def write_header(writer):
    """
    Write the header to the output CSV file. The header will include frequencies in the format f1, f1.5, f2, f2.5, ...
    """
    # frequencies = [f"f{i/4}" for i in range(4, 81)]  # Frequencies from 1 GHz to 20 GHz, every 0.5 GHz
    frequencies = [f"f{i/20}" for i in range(20, 401)]  # Frequencies from 1 GHz to 20 GHz, every 0.05 GHz
    header = ["Image Name"] + frequencies  # Add "Image Name" and frequency labels
    writer.writerow(header)


def process_input_csv(mode, input_file, output_file, parameter_names, param_order):
    """
    Process the input CSV file and generate the required output CSV format.
    """
    with open(input_file, 'r') as infile, open(output_file, mode, newline='') as outfile:
        reader = infile.readlines()
        writer = csv.writer(outfile)
        
        if mode == 'w':
            write_header(writer)

        image_filename = None
        db_values = []
        
        for line in reader:
            line = line.strip()
            
            # If the line contains the filter parameters, extract them
            if '/home/' in line:
                if image_filename and db_values:
                    # Write the previous image filename and corresponding DB values
                    writer.writerow([image_filename] + db_values)
                
                # Extract new image filename and parameters
                filename, params = extract_values_from_filename(line, parameter_names, param_order)
                image_filename = f'{filename}_{parameter_names[0]}{params[parameter_names[0]]}_{parameter_names[1]}{params[parameter_names[1]]}_{parameter_names[2]}{params[parameter_names[2]]}_{parameter_names[3]}{params[parameter_names[3]]}.png'
                db_values = []  # Reset DB values for the new filter
                
            # If the line contains frequency and DB value
            elif 'GHz' in line:
                continue  # Skip the header line (FREQUENCY (GHz), DB[S21]-...)
            else:
                # Split the frequency and DB value
                frequency, db_value = line.split(',')
                frequency = float(frequency)
                db_value = float(db_value)
                
                # Check if the frequency is a multiple of 0.5 GHz
                # if frequency % 0.25 == 0:
                    # db_values.append(db_value)
                db_values.append(db_value)
        
        # Don't forget to write the last set of data after the loop
        if image_filename and db_values:
            writer.writerow([image_filename] + db_values)

if __name__ == "__main__":
    # Input and output files
    output_file = './automation_scripts/annotations.csv'  # Replace with your desired output file path
    mode = 'a'
    # Process the input file and generate the output file
    process_input_csv('w', './automation_scripts/LPF_S21.csv', output_file, ['length', 'width', 'height', 'sep'], [1,4,2,3])
    process_input_csv(mode, './automation_scripts/HPF_S21.csv', output_file, ['width', 'width2', 'height', 'sep'], [2,4,3,1])
    process_input_csv(mode, './automation_scripts/BPF_S21.csv', output_file, ['length', 'length2', 'height', 'sep'], [1,3,2,4])
    process_input_csv(mode, './automation_scripts/notch_S21.csv', output_file, ['length', 'width', 'height', 'sep'], [1,2,3,4])