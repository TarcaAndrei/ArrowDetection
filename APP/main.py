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
inference_pipeline = InferencePipeline(device=device)


def detect_objects(image, clase_interes, confidenta_model, model_selector):
    inverse_decoder = {k:v for v, k in label_decoder.items()}
    clase_interes = [inverse_decoder[k] for k in clase_interes]
    imagine, predicted_labels = inference_pipeline.predict_one_image(image=image, confidenta=float(confidenta_model), clase_interes=clase_interes, model_type=model_selector)
    return imagine, predicted_labels

def detect_objects_in_video(video, clase_interes, confidenta_model, fps, model_selector):
    inverse_decoder = {k:v for v, k in label_decoder.items()}
    clase_interes = [inverse_decoder[k] for k in clase_interes]
    output_file, zip_path =  inference_pipeline.video_inference(video, batch_size=128, fps_final=int(fps), confidenta=float(confidenta_model), clase_interes=clase_interes, model_type=model_selector)
    return output_file, zip_path

def detect_objects_folders():
    pass

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
                model_selector = gr.Radio(["base", "small"], label="Select Model", value="small")
                input_clase_interes = gr.CheckboxGroup(choices=toate_clasele, label="Select classes", value=toate_clasele)
            with gr.Row():
                run_button_image = gr.Button("Run Detection")
            with gr.Row():
                output_image = gr.Image(type="numpy", label="Output Image")
            with gr.Row():
                output_json = gr.File(label="Download Labels JSON")  # New UI component for downloading
            run_button_image.click(fn=detect_objects, inputs=[input_image, input_clase_interes, confidence, model_selector], outputs=[output_image, output_json])
        
        with gr.Tab("Video Processing"):
            with gr.Row():
                gr.Markdown("<h2 style='text-align: center;'>Object Detection on Videos</h2>")
            with gr.Row():
                input_video = gr.Video()
            with gr.Accordion("Advanced options", open=False):
                confidence = gr.Slider(label="Confidence", minimum=0.0, maximum=1, value=0.8, step=0.05)
                fps = gr.Slider(label="FPS", minimum=5, maximum=60, value=15, step=5)
                model_selector = gr.Radio(["base", "small"], label="Select Model", value="small")
                input_clase_interes = gr.CheckboxGroup(choices=toate_clasele, label="Select classes", value=toate_clasele)
            with gr.Row():
                run_button_video = gr.Button("Run Detection")
            with gr.Row():
                output_video = gr.PlayableVideo(label="Output Video")
            with gr.Row():
                output_zip = gr.File(label="Download Annotated Images")  # New UI component for downloading
            run_button_video.click(fn=detect_objects_in_video, inputs=[input_video, input_clase_interes, confidence, fps, model_selector], outputs=[output_video, output_zip])

        # with gr.Tab("Folder Inference"):
        #     with gr.Row():
        #         gr.Markdown("<h2 style='text-align: center;'>Object Detection on Folder</h2>")
        #     with gr.Row():
        #         input_image = gr.Image(type="numpy")
        #     with gr.Accordion("Advanced options", open=False):
        #         confidence = gr.Slider(label="Confidence", minimum=0.0, maximum=1, value=0.9, step=0.05)
        #         height = gr.Number(label="Height", value=812, type='int')
        #         width = gr.Number(label="Width", value=1750, type='int')
        #         model_selector = gr.Radio(["base", "small"], label="Select Model", value="small")
        #         input_clase_interes = gr.CheckboxGroup(choices=toate_clasele, label="Select classes", value=toate_clasele)
        #     with gr.Row():
        #         run_button_image = gr.Button("Run Detection")
        #     with gr.Row():
        #         output_image = gr.Image(type="numpy", label="Output Image")
        #     with gr.Row():
        #         output_json = gr.File(label="Download Labels JSON")  # New UI component for downloading
        #     run_button_image.click(fn=detect_objects, inputs=[input_image, height, width, input_clase_interes, confidence, model_selector], outputs=[output_image, output_json])

# if __name__ == "__main__":
block.launch()
