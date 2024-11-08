import csv
import os

def process_csv(input_csv, output_txt):
    # Read the first line to extract the necessary parameters (image name and parameters)
    with open(input_csv, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        
        # Read the first row (which contains the metadata)
        first_line = next(reader)

        # Extract image name and parameters from the first line
        path_info = first_line[0].split()  # Split by space
        image_name = path_info[0].split('/')[-1].replace('.son', '')  # Image name without the '.son' extension


        # Extract parameters
        param_info = path_info[2:]  # Skip 'DE_EMBEDDED' and get parameter name-value pairs
        params = {}
        for param in param_info:
            param_name, param_value = param.split("=")
            params[param_name] = param_value
        
        # Initialize a list to store selected values for output
        selected_data = []
        
        # Read each row in the CSV
        for row in reader:
            try:
                frequency = float(row[0])  # Get the frequency value
                db_value = float(row[1])  # Get the DB[S21] value
                
                # Check if the frequency is a multiple of 0.5
                if frequency % 0.5 == 0:
                    selected_data.append(db_value)
            except ValueError:
                # Skip rows with invalid data
                continue
        
        # Format the output line based on the extracted parameters and data
        if selected_data:
            # Format the image filename with parameters
            output_image_name = f"{image_name}_L{params.get('L', '0')}_W{params.get('W', '0')}.png"
            
            # Create the output line
            output_line = f"{output_image_name}, " + ", ".join(map(str, selected_data))
            
            # Write the output line to the text file
            with open(output_txt, mode='a') as outputfile:
                outputfile.write(output_line + "\n")


for filename in os.listdir('./'):
    if filename.endswith('.csv'):
        process_csv(filename, './output.csv')
