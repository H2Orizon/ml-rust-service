import torch
from train import SentimentLSTM

model = SentimentLSTM()
model.load_state_dict(torch.load("model.pt"))
model.eval()

dummy_input = torch.randint(0, 10000, (1, 100))

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names = ["input"],
    output_names = ["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)