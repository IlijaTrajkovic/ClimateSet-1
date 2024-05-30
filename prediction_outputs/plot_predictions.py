import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_plot_predictions(directory):
    # Get a list of .npy files in the directory
    prediction_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
    # Load and plot each file
    for file in prediction_files:
        data = np.load(os.path.join(directory, file))
        # Assuming data shape is (12, 2, 96, 144) and you want to plot the first feature of the first data point
        plt.figure()
        plt.imshow(data[0, 0, :, :])  # Plotting the first 2D slice of the multidimensional array
        plt.colorbar()  # Add a color bar to clarify the scale
        plt.title(f'Plot of {file} - First Feature, First Data Point')
        plt.show()

# Usage
directory = '/Users/ilijatrajkovic/ClimateSet/prediction_outputs'
load_and_plot_predictions(directory)
