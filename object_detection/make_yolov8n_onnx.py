import torch
from ultralytics import YOLO

torch_model = YOLO("yolov8n.pt").model
torch_model.eval()

sample_input = torch.zeros(1, 3, 640, 640)

torch.onnx.export(
        torch_model, sample_input, "yolov8n.onnx", opset_version = 13, input_names = ["images"], output_names = ["outputs"]
)
