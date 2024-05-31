import hydra
<<<<<<< Updated upstream
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
import torch  # If needed for tensor handling
import numpy as np  # If needed for array handling
=======
from omegaconf import DictConfig, OmegaConf, ListConfig
import os
import numpy as np
import matplotlib.pyplot as plt
>>>>>>> Stashed changes
from emulator.src.utils.interface import get_model_and_data
from emulator.train import predict_model

@hydra.main(config_path="configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig):
    model, data = get_model_and_data(config)  # Loading the model and data
    predict_model(model, data, config)  # Prediction function
<<<<<<< Updated upstream
    print(model.predictions)
    output_dir = 'prediction_outputs'
    # Assuming model.predictions is a list of numpy arrays
    for idx, prediction in enumerate(model.predictions):
        file_path = os.path.join(output_dir, f'prediction_{idx}.npy')
        np.save(file_path, prediction)  # Saves each prediction as a .npy file in the directory
=======

    """# Average the predictions over the time-axis (axis=1)
    #averaged_predictions = [np.mean(prediction, axis=1) for prediction in model.predictions]"""

    # Assuming each prediction has shape [num_time_steps, num_vars , lat, lon]
    averaged_predictions = [np.mean(prediction[i], axis=0) for prediction in model.predictions for i in range(prediction.shape[0])]

    output_dir = 'prediction_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (avg_prediction, metadata) in enumerate(zip(averaged_predictions, model.predictions_metadata)):
        # Save averaged prediction
        file_path = os.path.join(output_dir, f'prediction_{idx}.npy')
        np.save(file_path, avg_prediction)

        # Plot the averaged prediction
        plot_prediction(avg_prediction, metadata, idx, output_dir)

def plot_prediction(prediction, metadata, idx, output_dir):
    output_variables = metadata['output_variable']
    # Ensure output_variables is a flat list
    if isinstance(output_variables, ListConfig):
        output_variables = list(output_variables)

    num_vars = prediction.shape[0]
    if len(output_variables) < num_vars:
        output_variables = (output_variables * ((num_vars // len(output_variables)) + 1))[:num_vars]


    for var_idx in range(prediction.shape[0]):  # Assuming prediction shape is [num_vars, lat, lon]
        plt.figure(figsize=(10, 6))
        vmin, vmax = determine_colorbar_limits(output_variables[var_idx])  # Call with specific variable
        display_label = get_display_label(output_variables[var_idx])
        im = plt.imshow(prediction[var_idx, :, :], cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label=f'{display_label} value')
        plt.title(f"Prediction {idx} - {display_label} - {metadata['scenario'][0]} - {metadata['year'][0].item()}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plot_path = os.path.join(output_dir, f"prediction_{display_label}_var_{metadata['year'][0].item()}.png")
        plt.savefig(plot_path)
        plt.close()

def determine_colorbar_limits(variable):
    # Directly determine the colorbar limits based on the specific variable type
    if 'tas' in variable:
        return -3, 3  # Temperature range in °C
    elif 'pr' in variable:
        return -3, 3  # Precipitation range in mm
    return None, None  # Default case if variable type is not recognized


def get_display_label(variable):
    if 'tas' in variable:
        return 'Temperature'
    elif 'pr' in variable:
        return 'Precipitation'
    return 'Variable'

>>>>>>> Stashed changes

if __name__ == "__main__":
    main()
