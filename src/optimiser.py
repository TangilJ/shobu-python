import logging
import os
from datetime import datetime

from ax.service.ax_client import AxClient, ObjectiveProperties

from src.environment import ModelConfig, Environment, EnvConfig

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
                    "bounds": [0.000001, 0.1],
                    "value_type": "float",
                    "log_scale": True,
                },
                {
                    "name": "c_puct",
                    "type": "range",
                    "bounds": [0.1, 3],
                    "value_type": "float",
                    "log_scale": False,
                },
                {
                    "name": "hidden_size",
                    "type": "range",
                    "bounds": [1, 100],
                    "value_type": "int",
                    "log_scale": False,
                },
                {
                    "name": "num_residual_blocks",
                    "type": "range",
                    "bounds": [1, 20],
                    "value_type": "int",
                    "log_scale": False,
                },
                {
                    "name": "policy_hidden_size",
                    "type": "range",
                    "bounds": [1, 100],
                    "value_type": "int",
                    "log_scale": False,
                },
                {
                    "name": "value_hidden_size",
                    "type": "range",
                    "bounds": [1, 100],
                    "value_type": "int",
                    "log_scale": False,
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
    def load(self, filename: str):
        ax = AxClient.load_from_json_file(filename)
        logger.info(f"Loaded optimiser from: {filename}")
        return Optimiser(ax)

    @staticmethod
    def _run_trial(p: dict) -> float:
        model = ModelConfig(
            hidden_size=p["hidden_size"],
            policy_hidden_size=p["policy_hidden_size"],
            num_residual_blocks=p["num_residual_blocks"],
            value_hidden_size=p["value_hidden_size"],
        )

        config = EnvConfig(
            learning_rate=p["lr"],
            c_puct=p["c_puct"],
            batch_size=64,
            iterations=5,
            playouts=1,
            epochs=20,
            simulations=25,
        )

        env = Environment.from_config(model, config)
        env.learn()

        loss = env.evaluate()
        return loss

    def run(self, trials: int):
        for trial in range(trials):
            parameters, trial_index = self.ax.get_next_trial()
            self.ax.complete_trial(
                trial_index=trial_index,
                raw_data=self._run_trial(parameters),
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

    optimiser = Optimiser.new()
    optimiser.run(20)
    optimiser.save()

    print(optimiser.ax.get_trials_data_frame())
