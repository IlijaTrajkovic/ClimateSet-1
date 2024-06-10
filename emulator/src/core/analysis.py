import numpy as np

class PredictionMetrics:
    def __init__(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

    def calculate_mse(self):
        """Calculate Mean Squared Error (MSE) between predictions and targets."""
        return np.mean(np.square(self.predictions - self.targets))

    def calculate_llrmse(self):
        """
        Calculate Log-likelihood Root Mean Squared Error (LLRMSE) between predictions and targets.
        Adjustments are made to avoid -inf and inf outcomes from log calculations.
        """
        # Ensure values are greater than -1 to safely apply np.log1p (log(1+x))
        predictions = np.clip(self.predictions, a_min=0, a_max=None)
        targets = np.clip(self.targets, a_min=0, a_max=None)

        # Handle NaNs and infs by replacing them with the median of the remaining valid values
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            valid_pred = predictions[np.isfinite(predictions)]
            median_pred = np.median(valid_pred) if valid_pred.size > 0 else 0
            predictions = np.nan_to_num(predictions, nan=median_pred, posinf=median_pred, neginf=median_pred)

        if np.any(np.isnan(targets)) or np.any(np.isinf(targets)):
            valid_targ = targets[np.isfinite(targets)]
            median_targ = np.median(valid_targ) if valid_targ.size > 0 else 0
            targets = np.nan_to_num(targets, nan=median_targ, posinf=median_targ, neginf=median_targ)

        # Compute log1p of clipped and cleaned predictions and targets
        log_predictions = np.log1p(predictions)
        log_targets = np.log1p(targets)

        # Calculate LLRMSE
        llrmse = np.sqrt(np.mean(np.square(log_predictions - log_targets)))

        return llrmse



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

