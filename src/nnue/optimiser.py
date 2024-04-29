import logging
import os
from datetime import datetime

from ax.service.ax_client import AxClient, ObjectiveProperties

from src.nnue.network import ModelConfig
from src.nnue.train import (
    TrainingConfig,
    EncodedData,
    evaluate,
    load_training_data,
    split_data,
    train,
)

logger = logging.getLogger(__name__)


class Optimiser:
    def __init__(self, ax: AxClient):
        self.ax = ax

    @staticmethod
    def new():
        ax = AxClient()
        ax.create_experiment(
            name="optimise",
            parameters=[
                {
                    "name": "lr",
                    "type": "range",
                    "bounds": [0.0000001, 0.1],
                    "value_type": "float",
                    "log_scale": True,
                },
                {
                    "name": "hidden_1",
                    "type": "choice",
                    "values": [32, 64, 128, 256, 512],
                },
                {
                    "name": "hidden_2",
                    "type": "choice",
                    "values": [32, 64, 128, 256, 512],
                },
                {
                    "name": "hidden_3",
                    "type": "choice",
                    "values": [32, 64, 128, 256, 512],
                },
            ],
            objectives={"loss": ObjectiveProperties(minimize=True)},
        )
        return Optimiser(ax)

    def save(self):
        t = datetime.now().strftime("%H%M%S")
        if "optimiser" not in os.listdir():
            os.mkdir("optimiser")
        self.ax.save_to_json_file(f"optimiser/{t}.json")
        logger.info(f"Saved optimiser to: optimiser/{t}.json")

    @staticmethod
    def load(filename: str):
        ax = AxClient.load_from_json_file(filename)
        logger.info(f"Loaded optimiser from: {filename}")
        return Optimiser(ax)

    @staticmethod
    def _run_trial(
        p: dict, train_data: [EncodedData], test_data: [EncodedData]
    ) -> float:
        model_config = ModelConfig(
            hidden_1=p["hidden_1"],
            hidden_2=p["hidden_2"],
            hidden_3=p["hidden_3"],
        )

        training_config = TrainingConfig(
            learning_rate=p["lr"],
            batch_size=1024,
            epochs=100,
            print_loss_step=25,
        )

        model = train(train_data, training_config, model_config)
        return evaluate(model, test_data, training_config.batch_size)

    def run(self, trials: int, train: [EncodedData], test: [EncodedData]):
        for trial in range(trials):
            parameters, trial_index = self.ax.get_next_trial()
            self.ax.complete_trial(
                trial_index=trial_index,
                raw_data=self._run_trial(parameters, train, test),
            )
            logger.info(f"Trial {trial + 1}/{trials} complete")

        logger.info(f"Optimisation complete. Best parameters:")
        best_parameters, _ = self.ax.get_best_parameters()
        logger.info(best_parameters)


if __name__ == "__main__":
    import pandas as pd

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    data = load_training_data("training_data.pkl")
    train_data, test_data = split_data(data, 0.8)

    optimiser = Optimiser.new()
    optimiser.run(20, train_data, test_data)
    optimiser.save()

    print(optimiser.ax.get_trials_data_frame())
