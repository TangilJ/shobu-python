import os
import pickle
import logging
import random
from dataclasses import dataclass
from datetime import datetime

import torch
from torch import nn

from src.nnue.network import NNUE, ModelConfig

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    learning_rate: float
    epochs: int
    batch_size: int
    print_loss_step: int


@dataclass
class EncodedData:
    board: torch.Tensor
    value: float


def load_training_data(path: str) -> [EncodedData]:
    with open(path, "rb") as f:
        data = pickle.load(f)
        logger.info(f"Read data from: {path}")
    return data


def save_checkpoint(model: NNUE, m_config: ModelConfig, t_config: TrainingConfig):
    if "models" not in os.listdir():
        os.mkdir("models")

    t = datetime.now().strftime("%H%M%S")
    path = f"models/nnue-{m_config.hidden_1}-{m_config.hidden_2}-{m_config.hidden_3}-{t_config.learning_rate}-{t}.pt"
    torch.save(model.state_dict(), path)
    logger.info(f"Saved model to: {path}")


def load_checkpoint(path: str, config: ModelConfig) -> NNUE:
    model = NNUE(config)
    model.load_state_dict(torch.load(path))
    model.eval()
    logger.info(f"Loaded model from: {path}")
    return model


def split_data(data: [EncodedData], split: float) -> ([EncodedData], [EncodedData]):
    random.shuffle(data)
    split_index = int(len(data) * split)
    return data[:split_index], data[split_index:]


def train(data: [EncodedData], config: TrainingConfig, model_config: ModelConfig):
    logger.info("Training model")
    model = NNUE(model_config)
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mse = nn.MSELoss()

    for epoch in range(config.epochs):
        running_loss = 0
        for i in range(0, len(data), config.batch_size):
            batch = data[i : i + config.batch_size]
            boards = torch.stack([x.board for x in batch])
            boards = boards.to(device=device, dtype=torch.float32)
            values = torch.tensor([x.value for x in batch])
            values = values.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(boards).squeeze()
            loss = mse(output, values.squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % config.print_loss_step == 0:
            running_loss /= len(data)
            logger.info(
                f"Finished epoch {epoch + 1}/{config.epochs} with loss: {running_loss}"
            )

    save_checkpoint(model, model_config, config)
    logger.info("Finished training")

    return model


def evaluate(model: NNUE, data: [EncodedData], batch_size: int) -> float:
    model = model.to(device=device)
    mse = nn.MSELoss()
    logger.info("Evaluating model")
    loss = 0
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        boards = torch.stack([x.board for x in batch])
        boards = boards.to(device=device, dtype=torch.float32)
        values = torch.tensor([x.value for x in batch])
        values = values.to(device=device, dtype=torch.float32)

        output = model(boards).squeeze()
        loss += mse(output, values).sum()

    final_loss = loss / len(data)
    logger.info(f"Finished evaluation. Loss: {final_loss}. Data size: {len(data)}")
    return final_loss.item()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    data = load_training_data("training_data.pkl")
    train_data, test_data = split_data(data, 0.8)

    model_config = ModelConfig(
        hidden_1=256,
        hidden_2=256,
        hidden_3=256,
    )
    training_config = TrainingConfig(
        learning_rate=0.0001,
        epochs=10,
        batch_size=1024,
        print_loss_step=10,
    )

    model = train(train_data, training_config, model_config)
    loss = evaluate(model, test_data, training_config.batch_size)

    logger.info(f"Loss: {loss}")
