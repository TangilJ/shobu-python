import logging
import os
from datetime import datetime

from ax.service.ax_client import AxClient, ObjectiveProperties

from src.environment import ModelConfig, Environment, EnvConfig

logger = logging.getLogger(__name__)


class Optimiser:
    def __init__(self):
        self.ax = AxClient()

        self.ax.create_experiment(
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

    def save(self):
        t = datetime.now().strftime("%H%M%S")
        if "optimiser" not in os.listdir():
            os.mkdir("optimiser")
        self.ax.save_to_json_file(f"optimiser/{t}.json")

    @staticmethod
    def load(filename: str):
        ax = AxClient()
        ax.load_from_json_file(filename)
        return ax

    @staticmethod
    def _run_trial(p: dict) -> float:
        model = ModelConfig(
            hidden_size=p["hidden_size"],
            policy_hidden_size=p["policy_hidden_size"],
            num_residual_blocks=p["num_residual_blocks"],
            value_hidden_size=p["value_hidden_size"],
            kernel_size=3,
        )

        config = EnvConfig(
            learning_rate=p["lr"],
            c_puct=p["c_puct"],
            batch_size=64,
            iterations=5,
            playouts=3,
            epochs=20,
            simulations=10,
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


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    optimiser = Optimiser()
    optimiser.run(20)
    optimiser.save()
