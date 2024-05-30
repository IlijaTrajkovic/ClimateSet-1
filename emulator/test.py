import hydra
from omegaconf import DictConfig, OmegaConf
from emulator.src.utils.interface import get_model_and_data
from emulator.train import test_model

@hydra.main(config_path="configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig):
    # Load the pre-trained model
    model, data = get_model_and_data(config)

    # Call the test_model function with the loaded model
    test_model(model, data, config)


if __name__ == "__main__":
    main()
