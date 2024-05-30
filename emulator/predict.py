import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
import torch  # If needed for tensor handling
import numpy as np  # If needed for array handling
from emulator.src.utils.interface import get_model_and_data
from emulator.train import predict_model

@hydra.main(config_path="configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig):
    model, data = get_model_and_data(config)  # Loading the model and data
    predict_model(model, data, config)  # Prediction function
    print(model.predictions)
    output_dir = 'prediction_outputs'
    # Assuming model.predictions is a list of numpy arrays
    for idx, prediction in enumerate(model.predictions):
        file_path = os.path.join(output_dir, f'prediction_{idx}.npy')
        np.save(file_path, prediction)  # Saves each prediction as a .npy file in the directory

if __name__ == "__main__":
    main()
