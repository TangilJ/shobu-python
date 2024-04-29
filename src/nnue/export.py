import torch

from src.nnue.network import NNUE, ModelConfig


def tensor_1d_to_string(tensor):
    return ",".join(str(x.item()) for x in tensor)


def tensor_2d_to_string(tensor):
    return ";".join(tensor_1d_to_string(row) for row in tensor)


def model_to_string(model):
    output = ""
    for attr in ["initial", "hidden_1", "hidden_2", "output"]:
        layer = getattr(model, attr)
        output += tensor_2d_to_string(layer.weight.data) + "\n"
        output += tensor_1d_to_string(layer.bias.data) + "\n"
    return output[:-1]  # Remove trailing newline


if __name__ == "__main__":
    model = NNUE(ModelConfig(512, 64, 64))
    model.load_state_dict(torch.load("models/nnue-512-64-64.pt"))
    model.eval()

    with open("models/512-64-64.nnue", "w") as f:
        f.write(model_to_string(model))
