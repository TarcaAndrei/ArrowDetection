import gradio as gr
from PIL import Image
import sys
sys.path.append("/teamspace/studios/this_studio/ArrowDetection")
sys.path.append("/teamspace/studios/this_studio/ArrowDetection/ML")
from ML import InferencePipeline, label_decoder
import torch
import time

toate_clasele = [v for _, v in label_decoder.items()]

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
inference_pipeline = InferencePipeline(weights="/teamspace/studios/this_studio/weights/model_base.pth",device=device, model_type="base")


def detect_objects(image, crop_height, crop_width, clase_interes, confidenta_model):
    inverse_decoder = {k:v for v, k in label_decoder.items()}
    clase_interes = [inverse_decoder[k] for k in clase_interes]
    return inference_pipeline.predict_one_image(image=image, end_height=int(crop_height), end_width=int(crop_width), confidenta=float(confidenta_model), clase_interes=clase_interes)

def detect_objects_in_video(video,crop_height, crop_width, clase_interes, confidenta_model, fps):
    inverse_decoder = {k:v for v, k in label_decoder.items()}
    print(clase_interes)
    clase_interes = [inverse_decoder[k] for k in clase_interes]
    output_file =  inference_pipeline.video_inference(video, batch_size=64, fps_final=int(fps), end_height=int(crop_height), end_width=int(crop_width), confidenta=float(confidenta_model), clase_interes=clase_interes)
    time.sleep(3)
    return output_file

block = gr.Blocks().queue()
with block:
    with gr.Tabs():
        with gr.Tab("Image Processing"):
            with gr.Row():
                gr.Markdown("<h2 style='text-align: center;'>Object Detection on Images</h2>")
            with gr.Row():
                input_image = gr.Image(type="numpy")
            with gr.Accordion("Advanced options", open=False):
                confidence = gr.Slider(label="Confidence", minimum=0.0, maximum=1, value=0.9, step=0.05)
                height = gr.Number(label="Height", value=812, type='int')
                width = gr.Number(label="Width", value=1750, type='int')
                input_clase_interes = gr.CheckboxGroup(choices=toate_clasele, label="Select classes", value=toate_clasele)
            with gr.Row():
                run_button_image = gr.Button("Run Detection")
            with gr.Row():
                output_image = gr.Image(type="numpy", label="Output Image")
            run_button_image.click(fn=detect_objects, inputs=[input_image, height, width, input_clase_interes, confidence], outputs=output_image)
        
        with gr.Tab("Video Processing"):
            with gr.Row():
                gr.Markdown("<h2 style='text-align: center;'>Object Detection on Videos</h2>")
            with gr.Row():
                input_video = gr.Video()
            with gr.Accordion("Advanced options", open=False):
                confidence = gr.Slider(label="Confidence", minimum=0.0, maximum=1, value=0.8, step=0.05)
                fps = gr.Slider(label="FPS", minimum=1, maximum=60, value=15, step=5)
                height = gr.Number(label="Height", value=812, type='int')
                width = gr.Number(label="Width", value=1750, type='int')
                input_clase_interes = gr.CheckboxGroup(choices=toate_clasele, label="Select classes", value=toate_clasele)
            with gr.Row():
                run_button_video = gr.Button("Run Detection")
            with gr.Row():
                output_video = gr.PlayableVideo(label="Output Video")
            run_button_video.click(fn=detect_objects_in_video, inputs=[input_video, height, width, input_clase_interes, confidence, fps], outputs=output_video)

# if __name__ == "__main__":
block.launch()
