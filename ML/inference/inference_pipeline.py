from inference.model_creator import build_inference_model
from inference.data_process import PreprocessData
from inference.plot_predictions import plot_prediction
from pathlib import Path
import torch


class Inference:
    def __init__(self, weights: str, device: torch.device) -> None:
        self.model = build_inference_model(weights=weights, device=device)
        self.device = device
        self.data_preprocessing = PreprocessData(device=device)
        self.input_folder = Path("/home/tan8clj/images/inputs")

    def inference(self, image_name):
        image_path = self.input_folder / image_name
        preprocessed_image = self.data_preprocessing.preprocess_image(
            image_path).unsqueeze(0)
        with torch.inference_mode():
            outputs = self.model(preprocessed_image)
            plot_prediction(preprocessed_image[0], outputs, image_name)


if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    weights = "weights/model_small.pth"
    inference = Inference(weights=weights, device=device)
    inference.inference("img_2.png")
