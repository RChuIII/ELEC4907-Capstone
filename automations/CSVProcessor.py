import csv
import re

def extract_values_from_filename(filepath):
    """
    Extract filter parameters (width, length, height) and construct the image filename.
    Example input: "/home/vrajshah4/Desktop/4907_Test/folded_double_stub_notch_filter_param.son DE_EMBEDDED width=20.0 length=190.0 height=22.0"
    Example output: ("folded_double_stub_notch_filter_param", "20", "190", "22")
    """
    # Extract width, length, and height using regex from the file path
    match = re.search(r'width=([0-9\.]+).*length=([0-9\.]+).*height=([0-9\.]+)', filepath)
    if match:
        width = match.group(1)[:-2]
        length = match.group(2)[:-2]
        height = match.group(3)[:-2]
        # Return the base filename and the extracted parameters
        base_filename = filepath.split('/')[-1].split(' ')[0].replace('.son', '')
        return base_filename, width, length, height
    return None, None, None, None

def write_header(writer):
    """
    Write the header to the output CSV file. The header will include frequencies in the format f1, f1.5, f2, f2.5, ...
    """
    # frequencies = [f"f{i/2}" for i in range(2, 41)]  # Frequencies from 1 GHz to 20 GHz, every 0.5 GHz
    frequencies = [f"f{i/20}" for i in range(20, 401)]  # Frequencies from 1 GHz to 20 GHz, every 0.05 GHz
    header = ["Image Name"] + frequencies  # Add "Image Name" and frequency labels
    writer.writerow(header)


def process_input_csv(input_file, output_file):
    """
    Process the input CSV file and generate the required output CSV format.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = infile.readlines()
        writer = csv.writer(outfile)
        
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
                filename, width, length, height = extract_values_from_filename(line)
                image_filename = f'{filename}_height{height}_length{length}_width{width}.png'
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
                # if frequency % 0.5 == 0:
                #     db_values.append(db_value)
                db_values.append(db_value)
        
        # Don't forget to write the last set of data after the loop
        if image_filename and db_values:
            writer.writerow([image_filename] + db_values)

if __name__ == "__main__":
    # Input and output files
    input_file = './3param_10p.csv'  # Replace with your actual input file path
    output_file = './annotations.csv'  # Replace with your desired output file path

    # Process the input file and generate the output file
    process_input_csv(input_file, output_file)
    