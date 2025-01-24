import gradio as gr
import os
import shutil
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import parameters
import validation
import training

def copy_files_from_temp_storage(folder: str, *files: gr.utils.NamedString) -> str:
    try:
        for file in files:
            if file is not None:
                file_name, file_extension = os.path.splitext(os.path.basename(file))
                if file_extension == '.pth':
                    shutil.copy(file, os.path.join('./data/models_folder', os.path.basename(file)))
                else:
                    shutil.copy(file, os.path.join(f'./data/uploads/{folder}', os.path.basename(file)))
        return 'Upload successful!' 

    except Exception as e:
        return f"An error occurred: {e}"

# Placeholder function for model training
def train_model( model_name, csv_file, zip_file, json_file):
    annotations = os.path.join('./data/uploads/Training', os.path.basename(csv_file))
    # parameters = os.path.join('./uploads/Training', os.path.basename(json_file))
    
    training_image_directory = './data/uploads/Training/test_images'
    empty_directory(training_image_directory)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(training_image_directory)
    
    training_time = training.training_loop(csv_file=annotations,
                           img_dir=training_image_directory,
                           model_name=model_name 
                           )
    shutil.move(f'./{model_name}.pth', f'./data/models_folder/{model_name}.pth')
    return f"Training completed in {training_time} seconds. Model saved as: {model_name}."


#------------------------------- Validation -------------------------------#
def empty_directory(directory_path: str) -> None:
    os.makedirs(directory_path, exist_ok=True)      # Create directory if it doesn't exist.
    if os.listdir(directory_path) is not None:      # If directory isn't empty...
        shutil.rmtree(directory_path)               # Remove it and all items in it.
        os.makedirs(directory_path, exist_ok=True)  # Re-create the directory.

def plot_output(points, start, end, num_points=parameters.num_outputs):
    # Generate x values based on the given range
    x_values = np.linspace(start, end, num_points)
    
    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, points, marker='o', linestyle='-', color='b')
    plt.title('Graph of 40 Points')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    
    # Save the plot as an image file
    plot_filename = 'output_plot.png'
    plot_filepath = os.path.join(os.getcwd(), plot_filename)
    plt.savefig(plot_filepath)
    plt.close()  # Close the plot to free memory

    # Return the file path of the saved image
    return plot_filepath

# Placeholder function for validation
def validate_model(pth_file: str, zip_file: str, single_test_image: str) -> list:
    test_image_directory = './data/uploads/Validation/test_images'
    empty_directory(test_image_directory)

    if zip_file is None and single_test_image is None:
        return  "Validation Failed. No Image(s)"

    if zip_file is not None:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(test_image_directory)
        test_image_file = validation.choose_random_file('./data/uploads/Validation/test_images')
    else:
        test_image_file = single_test_image
    prediction = validation.predict_output(f'./data/models_folder/{pth_file}', f'./data/uploads/Validation/test_images/{test_image_file}')
    plotted_image = plot_output(prediction, 0, 10)
    return f"Validation completed successfully (Validated {test_image_file} using {os.path.basename(pth_file)})!", plotted_image

# Function to list .pth files in the models directory
def list_models(models_dir):
    # return [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    return [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f))]

def update_model_choices(model_file):
    # Update the model dropdown with the new list of .pth files
    models = list_models("./data/models_folder")
    return gr.Dropdown(choices=models)

# Define Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# ELEC 4907 : Neural Networks for High-Frequency Electronic Modeling")
    
    with gr.Tabs():
        # Training Tab
        with gr.Tab("Training"):
            with gr.Row():
                with gr.Column(scale=1):
                    training_csv_upload = gr.File(label="Upload CSV (Annotations)")
                    training_zip_upload = gr.File(label="Upload ZIP (Image Dataset)")
                    training_json_upload = gr.File(label="Upload JSON (Model Parameters)")
                    training_model_name_input = gr.Textbox(label="Model Name", interactive=True)
                    training_upload_button = gr.Button("Upload Files")
                    training_folder_input = gr.Textbox(label="Target Folder Placeholder", placeholder="Training", interactive=True)
                with gr.Column(scale=1):
                    training_train_button = gr.Button("Train Model")
                    training_output_text = gr.Textbox(label="Output", interactive=False)

            training_upload_button.click(
                copy_files_from_temp_storage,
                inputs = [training_folder_input, training_csv_upload, training_zip_upload, training_json_upload],
                outputs = training_output_text
            )
            training_train_button.click(
                train_model,
                inputs = [training_model_name_input, training_csv_upload, training_zip_upload, training_json_upload],
                outputs = training_output_text
            )

        # Validation Tab
        with gr.Tab("Validation"):
            with gr.Row():
                with gr.Column(scale=1):
                    validation_model_selector = gr.Dropdown(label="Select Model", choices=list_models("./data/models_folder"), interactive=True)
                    validation_model_upload = gr.File(label="Upload pth model file (Model File)")
                    validation_zip_upload = gr.File(label="Upload ZIP (Test Images)")
                    validation_upload_button = gr.Button("Upload Files")
                    validation_folder_input = gr.Textbox(label="Target Folder Placeholder", placeholder="Validation", interactive=True)
                    validation_image_upload = gr.File(label="Upload png file (Single Test Image)")
                with gr.Column(scale=1):
                    validate_button = gr.Button("Test")
                    validation_output = gr.Textbox(label="Output", interactive=False)
                    validation_plot = gr.Image(label="Validation Results")

            validate_button.click(
                validate_model,
                inputs=[validation_model_selector, validation_zip_upload, validation_image_upload],
                outputs=[validation_output, validation_plot]
            )

            validation_upload_button.click(
                copy_files_from_temp_storage,
                inputs = [validation_folder_input, validation_model_upload],
                outputs = [training_output_text],
            ).then(
                update_model_choices,  # Trigger dropdown update after upload
                inputs=[validation_model_upload],
                outputs=[validation_model_selector]
            )

if __name__ == "__main__":
    # Empty and create directories
    os.makedirs('./data/models_folder', exist_ok=True)
    empty_directory('./data/uploads/Training')
    empty_directory('./data/uploads/Validation')

    # Launch the Gradio app
    app.launch()
