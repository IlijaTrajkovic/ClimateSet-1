import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
import torch  # If needed for tensor handling
import numpy as np  # If needed for array handling
from omegaconf import DictConfig, OmegaConf, ListConfig
import os
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from emulator.src.utils.interface import get_model_and_data
from emulator.train import predict_model

@hydra.main(config_path="configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig):
    model, data = get_model_and_data(config)  # Loading the model and data
    predict_model(model, data, config)  # Prediction function
    output_dir = 'prediction_outputs'

    # Assuming each prediction has shape [num_time_steps, num_vars , lat, lon]
    averaged_predictions = [np.mean(prediction[i], axis=0) for prediction in model.predictions for i in range(prediction.shape[0])]

    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (avg_prediction, metadata) in enumerate(zip(averaged_predictions, model.predictions_metadata)):
        # Save averaged prediction
        #file_path = os.path.join(output_dir, f'prediction_{idx}.npy')
        #np.save(file_path, avg_prediction)

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
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.coastlines(resolution='110m')
        ax.set_global()
        display_label = get_display_label(output_variables[var_idx])
        unit_label = get_units(output_variables[var_idx])
        cmap = 'coolwarm' if 'tas' in output_variables[var_idx] else 'Blues'
        data = np.squeeze(prediction[var_idx, :, :])
        im = ax.imshow(data, cmap=cmap, origin='lower', transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90], interpolation ='none')
        cbar = plt.colorbar(im, label=f'{display_label} ({unit_label})', orientation='vertical')
        cbar.ax.tick_params(labelsize=10)  # Adjust tick size on colorbar
        plt.title(f"{display_label} - {metadata['scenario'][0]} - {metadata['year'][0].item()}", fontweight='bold', fontsize=14)
        plot_path = os.path.join(output_dir, f"{display_label}_{metadata['year'][0].item()}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()


def get_display_label(variable):
    if 'tas' in variable:
        return 'Temperature'
    elif 'pr' in variable:
        return 'Precipitation'
    return 'Variable'

def get_units(variable):
    if 'tas' in variable:
        return 'K'  # Degrees Kelvin
    elif 'pr' in variable:
        return 'mm/day'  # Millimeters per day
    return 'units'  # Generic unit label if unknown

if __name__ == "__main__":
    main()
