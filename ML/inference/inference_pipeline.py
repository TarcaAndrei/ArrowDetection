from inference.model_creator import build_inference_model
from inference.data_process import PreprocessData
from inference.plot_predictions import plot_frame_prediction
from pathlib import Path
import os
import cv2
# to take all files from dir
from inference.clase import label_decoder
import time

import torch


class InferencePipeline:
    def __init__(self, weights: str, device: torch.device, model_type: str = "small") -> None:
        self.model = build_inference_model(
            weights=weights, device=device, model_type=model_type)
        self.device = device
        self.data_preprocessing = PreprocessData(device=device)

    def video_inference(self, video_name, fps_final=15, batch_size=32, end_height=812, end_width=1750, confidenta=0.0, clase_interes=label_decoder.keys()):
        cap = cv2.VideoCapture(video_name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_initial = int(cap.get(cv2.CAP_PROP_FPS)) + 1
        # Get total frame count
        total_frames_initial = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate video duration (in seconds)
        duration = total_frames_initial // fps_initial if fps_initial > 0 else 0
        print(f"Duration: {duration} ---- FPS: {fps_initial}")
        every_frame = fps_initial // fps_final
        output_video_path = "/teamspace/studios/this_studio/videos/output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # we need this to display video in browser
        out = cv2.VideoWriter(output_video_path, fourcc,
                              fps_final, (end_width, end_height))
        frame_count = 0
        frame_list = []
        numar_ploturi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if no more frames
            if frame_count % every_frame == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(frame)
                if len(frame_list) == batch_size:
                    numar_ploturi += 1
                    init_imgs, preprocessed_imgs = self.data_preprocessing.preprocess_sequences(
                        frame_list, end_height, end_width)
                    with torch.inference_mode():
                        outputs = self.model(preprocessed_imgs)
                        for (img, pred_box, pred_log) in zip(init_imgs, outputs["pred_boxes"], outputs["pred_logits"]):
                            tmp_out = {
                                "pred_boxes": pred_box,
                                "pred_logits": pred_log,
                            }
                            returned_frame = plot_frame_prediction(
                                img, tmp_out, save=False, confidenta=confidenta, clase_interes=clase_interes)
                            out.write(returned_frame)
                        print(f"Plotted a batch of images...{numar_ploturi}")
                    frame_list = []
                    if numar_ploturi == 5:
                        break
            frame_count += 1
        cap.release()
        if len(frame_list) > 0:
            init_imgs, preprocessed_imgs = self.data_preprocessing.preprocess_sequences(
                frame_list, end_height, end_width)
            with torch.inference_mode():
                outputs = self.model(preprocessed_imgs)
                for (img, pred_box, pred_log) in zip(init_imgs, outputs["pred_boxes"], outputs["pred_logits"]):
                    tmp_out = {
                        "pred_boxes": pred_box,
                        "pred_logits": pred_log,
                    }
                    returned_frame = plot_frame_prediction(
                        img, tmp_out, save=False, confidenta=confidenta, clase_interes=clase_interes)
                    out.write(returned_frame)
        out.release()
        return output_video_path

    def folder_inference(self, folder_name, output_dir, batch_size=32, end_height=812, end_width=1750, confidenta=0.0, clase_interes=label_decoder.keys()):
        all_imgs = os.listdir(folder_name)
        all_imgs.sort()
        os.makedirs(output_dir, exist_ok=True)
        numar_frame = 0
        for i in range(0, len(all_imgs), batch_size):
            batch_imagini = all_imgs[i: i+batch_size]
            if len(batch_imagini) == 0:
                break
            initial_imgs, preprocessed_imgs = self.data_preprocessing.preprocess_batch_images(
                folder_name, batch_imagini, end_height=end_height, end_width=end_width)
            with torch.inference_mode():
                outputs_imgs = self.model(preprocessed_imgs)
                for (img, pred_box, pred_log) in zip(initial_imgs, outputs_imgs["pred_boxes"], outputs_imgs["pred_logits"]):
                    numar_frame += 1
                    tmp_out = {
                        "pred_boxes": pred_box,
                        "pred_logits": pred_log,
                    }
                    plot_frame_prediction(img, tmp_out, f"Frame_{numar_frame}", output_dir=output_dir, save=True,
                                          clase_interes=clase_interes, confidenta=confidenta)
            print(f"Plotted a batch of images!")


    def one_image_inference(self, input_folder, image_name, output_dir, end_height=812, end_width=1750, confidenta=0.0, clase_interes=label_decoder.keys()):
        os.makedirs(output_dir, exist_ok=True)
        initial_image, preprocessed_image = self.data_preprocessing.preprocess_batch_images(input_folder, [image_name], end_height=end_height, end_width=end_width)
        with torch.inference_mode():
            outputs = self.model(preprocessed_image)
            plot_frame_prediction(initial_image[0], outputs, f"Inf_{image_name}", output_dir=output_dir, save=True,
                                          clase_interes=clase_interes, confidenta=confidenta)
            print("Image plotted!")

    def predict_one_image(self, image, end_height=812, end_width=1750, confidenta=0.0, clase_interes=label_decoder.keys()):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, preprocessed_img = self.data_preprocessing.preprocess_sequences([image], end_height, end_width)
        # initial_image, preprocessed_image = self.data_preprocessing.preprocess_batch_images(input_folder, [image_name], end_height=end_height, end_width=end_width)
        with torch.inference_mode():
            outputs = self.model(preprocessed_img)
            img_with_bbox = plot_frame_prediction(image[0], outputs, save=False, clase_interes=clase_interes, confidenta=confidenta)
            print("Image plotted!")
        return img_with_bbox


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    weights = "/teamspace/studios/this_studio/weights/model_base.pth"
    inference = InferencePipeline(
        weights=weights, device=device, model_type="base")
    clase_de_interes = label_decoder.keys()
    inference.video_inference("video.mp4",
                              fps_final=6, confidenta=0.9, clase_interes=clase_de_interes)
    inference.folder_inference("input", output_dir="output")
    inference.one_image_inference("input", "name.jpg", output_dir="output")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    # 340 secunde - video de 1 minut 60fps -> 1 minut 15 fps ()
