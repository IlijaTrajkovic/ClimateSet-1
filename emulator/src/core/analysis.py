import numpy as np

class PredictionMetrics:
    def __init__(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

    def calculate_mse(self):
        """Calculate Mean Squared Error (MSE) between predictions and targets."""
        return np.mean(np.square(self.predictions - self.targets))

    def calculate_llrmse(self):
        """Calculate Log-likelihood Root Mean Squared Error (LLRMSE) between predictions and targets."""
        return np.sqrt(np.mean(np.square(np.log1p(self.predictions) - np.log1p(self.targets))))

    def custom_metric_1(self):
        """Implement your custom metric 1."""
        # Dummy implementation
        value = np.mean(self.predictions - self.targets)  # Example calculation
        return value

    def custom_metric_2(self):
        """Implement your custom metric 2."""
        # Dummy implementation
        value = np.median(self.predictions - self.targets)  # Example calculation
        return value

    def calculate_average_errors_over_time(self):
        """
        Calculate average errors over time from gathered prediction data.

        Args:
            gathered_data (dict): Dictionary containing gathered prediction data.

        Returns:
            dict: Dictionary containing calculated average errors.
        """

        mse = self.calculate_mse()
        llrmse = self.calculate_llrmse()

        average_errors = {
            "mse": mse,
            "llrmse": llrmse,
        }
        return average_errors

    def perform_analysis(self, metrics):
        """
        Perform analysis on model predictions based on selected metrics.
        
        Args:
            metrics (list): List of strings specifying which metrics to calculate.

        Returns:
            dict: Dictionary containing calculated metrics.
        """
        calculated_metrics = {}
        for metric in metrics:
            if metric == "mse":
                calculated_metrics["mse"] = self.calculate_mse()
            elif metric == "llrmse":
                calculated_metrics["llrmse"] = self.calculate_llrmse()
            elif metric == "custom_metric_1":
                calculated_metrics["custom_metric_1"] = self.custom_metric_1()
            elif metric == "custom_metric_2":
                calculated_metrics["custom_metric_2"] = self.custom_metric_2()
        return calculated_metrics

