import logging

from network import AlphaZero
from src.environment import Environment, EnvConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger.info("Starting")

POLICY_OUTPUT_SIZE = (
    16  # passive_source_index
    * 16  # aggressive_source_index
    * 2  # passive_side
    * 2  # aggressive_side
    * 8  # direction
    * 2  # times_moved
)  # = 16384

model = AlphaZero(
    hidden_size=5,
    policy_hidden_size=5,
    policy_output_size=POLICY_OUTPUT_SIZE,
    num_residual_blocks=3,
)


config = EnvConfig(
    policy_size=POLICY_OUTPUT_SIZE,
    learning_rate=0.001,
    c_puct=1,
    batch_size=64,
    iterations=5,
    playouts=10,
    epochs=20,
    simulations=25,
)

logger.info("Creating environment")
env = Environment(model, config)
env.learn()
