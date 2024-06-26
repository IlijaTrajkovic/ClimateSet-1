import wandb
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import emulator.src.utils.config_utils as cfg_utils
from emulator.src.utils.interface import get_model_and_data
from emulator.src.utils.utils import get_logger


def run_model(config: DictConfig):
    seed_everything(config.seed, workers=True)
    log = get_logger(__name__)
    log.info("In run model")
    cfg_utils.extras(config)

    log.info("Running model")
    if config.get("print_config"):
        cfg_utils.print_config(config, fields="all")

    emulator_model, data_module = get_model_and_data(config)
    log.info("Got model")

    # Init Lightning callbacks and loggers
    callbacks = cfg_utils.get_all_instantiable_hydra_modules(config, "callbacks")
    loggers = cfg_utils.get_all_instantiable_hydra_modules(config, "logger")

    # Init Lightning trainer
    trainer: pl.Trainer = hydra_instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=loggers,  # , deterministic=True
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters to the PyTorch Lightning loggers.")
    cfg_utils.log_hyperparameters(
        config=config,
        model=emulator_model,
        data_module=data_module,
        trainer=trainer,
        callbacks=callbacks,
    )

    trainer.fit(model=emulator_model, datamodule=data_module)

    cfg_utils.save_hydra_config_to_wandb(config)

    if config.get("test_after_training"):
        trainer.test(datamodule=data_module, ckpt_path="best")

    if config.get("logger") and config.logger.get("wandb"):
        wandb.finish()


#Test model function that directly tests an already trained model
def test_model(model, data, config: DictConfig):
    log = get_logger(__name__)
    log.info("In test mode")
    cfg_utils.extras(config)

    log.info("Testing model")
    if config.get("print_config"):
        cfg_utils.print_config(config, fields="all")

    print(data)
    trainer = pl.Trainer(logger=False)
    trainer.test(model=model, datamodule=data) #ckpt-best needed???

    if config.get("logger") and config.logger.get("wandb"):
        wandb.finish()

#Predict model function that directly tests an already trained model
def predict_model(model, data, config: DictConfig):
    log = get_logger(__name__)
    log.info("In predict mode")
    cfg_utils.extras(config)

    log.info("Predicting model")
    if config.get("print_config"):
        cfg_utils.print_config(config, fields="all")

    loggers = cfg_utils.get_all_instantiable_hydra_modules(config, "logger")

    # Init Lightning trainer
    trainer: pl.Trainer = hydra_instantiate(
        config.trainer,
        logger=loggers
    )
    print("got trainer")
    trainer.predict(model=model, datamodule=data) #ckpt-best needed???

    if config.get("logger") and config.logger.get("wandb"):
        wandb.finish()

