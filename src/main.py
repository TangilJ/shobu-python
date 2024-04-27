import logging

from src.environment import Environment, EnvConfig, ModelConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger.info("Starting")


model = ModelConfig(
    hidden_size=5,
    policy_hidden_size=5,
    value_hidden_size=5,
    num_residual_blocks=3,
)


config = EnvConfig(
    learning_rate=0.001,
    c_puct=1,
    batch_size=64,
    iterations=5,
    playouts=10,
    epochs=20,
    simulations=25,
)

logger.info("Creating environment")
env = Environment.from_config(model, config)
env.learn()
logger.info("Finished training")

losses = []
for _ in range(5):
    loss = env.evaluate()
    losses.append(loss)
    logger.info(f"Loss: {loss}")

logger.info(f"Losses: {losses}")
logger.info(f"Average loss: {sum(losses) / len(losses)}")