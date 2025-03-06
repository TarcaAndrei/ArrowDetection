from inference.model_creator import build_inference_model
from inference.data_process import PreprocessData
from inference.plot_predictions import plot_prediction
from pathlib import Path
from torchinfo import summary
import torch


class Inference:
    def __init__(self, weights: str, device: torch.device, model_type: str = "small") -> None:
        self.model = build_inference_model(weights=weights, device=device, model_type=model_type)
        self.device = device
        self.data_preprocessing = PreprocessData(device=device)
        self.input_folder = Path("<input_folder>")

    def inference(self, image_name, output_dir):
        image_path = self.input_folder / image_name
        preprocessed_image = self.data_preprocessing.preprocess_image(
            image_path).unsqueeze(0)
        with torch.inference_mode():
            outputs = self.model(preprocessed_image)
            plot_prediction(preprocessed_image[0], outputs, image_name, output_dir=output_dir)


if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:7")
    weights = "<path_to_weights>"
    inference = Inference(weights=weights, device=device, model_type="small")
    # print(summary(inference.model, input_size=(1, 3, 518, 1666)))
    inference.inference("img.jpg", output_dir="<output_dir>")
