import gradio as gr
import cv2
import numpy as np
import pandas as pd
import yolov5
import os

# Loading the fine-tuned YOLOv5 model
model = yolov5.load('runs/train/exp4/weights/best.pt')

# Preprocessing to improve detection (resize, denoise)
def preprocess_image(img):
    img = cv2.resize(img, (640, 640))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# Gap detection using YOLO
def detect_gaps(image):
    results = model(image)
    preds = results.pred[0]
    gap_boxes = [list(map(int, box[:4])) for box in preds]
    return gap_boxes, results

# Image quality using histogram contrast (darkness) and Laplacian variance (blurriness)
def image_quality_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255  # Normalized brightness
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(lap_var / 1000, 1.0)  # Normalize
    score = round((0.5 * brightness + 0.5 * sharpness) * 100, 2)
    return score

# Gap density = num gaps / image area (normalized)
def calculate_gap_density(gap_boxes, image_shape):
    img_area = image_shape[0] * image_shape[1]
    gap_area = sum([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in gap_boxes])
    density = gap_area / img_area
    return float(round(density, 4))

def compute_compliance_score(gap_score, quality_score, density_score):
    return round(0.5 * gap_score + 0.3 * quality_score + 0.2 * density_score, 2)

def analyze_image(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    preprocessed = preprocess_image(img.copy())

    gap_boxes, results = detect_gaps(preprocessed)
    quality = image_quality_score(preprocessed)
    density = calculate_gap_density(gap_boxes, preprocessed.shape)

    # Gap score (inverse of density): high density = bad
    gap_score = round((1 - float(density)) * 100, 2)       # Higher = better
    gap_density_score = round(float(density) * 100, 2)     # Higher = worse
    compliance_score = compute_compliance_score(gap_score, quality, gap_density_score)

    # Draw bounding boxes
    for x1, y1, x2, y2 in gap_boxes:
        cv2.rectangle(preprocessed, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Convert for Gradio display
    img_out = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

    score_breakdown = pd.DataFrame([
        ["Gap Score (50%)", f"{gap_score}%"],
        ["Image Quality (30%)", f"{quality}%"],
        ["Gap Density (20%)", f"{gap_density_score}%"],
    ], columns=["Component", "Score"])

    final_score_text = f"{compliance_score}%"

    return img_out, final_score_text, score_breakdown

# Image options from test folder
image_dir = "datasets/shelf_planograms/DATASET_Planogram/test/images"
def get_image_list():
    return [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg'))]

dropdown = gr.Dropdown(choices=["Select"] + get_image_list(), label="Select Shelf Image from Test Set")
upload = gr.Image(type="pil", label="Or Upload an Image")
output_img = gr.Image(label="Annotated Output")
output_score = gr.Textbox(label="Compliance Score")
output_json = gr.Dataframe(headers=["Component", "Score"], label="Score Breakdown by Component")

def image_selector(dropdown_image, uploaded_image):
    if uploaded_image is not None:
        return uploaded_image
    elif dropdown_image:
        return cv2.cvtColor(cv2.imread(os.path.join(image_dir, dropdown_image)), cv2.COLOR_BGR2RGB)
    return None

interface = gr.Interface(
    fn=lambda dropdown, upload: analyze_image(image_selector(dropdown, upload)),
    inputs=[dropdown, upload],
    outputs=[output_img, output_score, output_json],
    title="Retail Shelf Gap & Compliance Analyzer",
    description="<div style='text-align:center'>Detects shelf gaps, scores image quality, calculates gap density, and computes a final compliance score.</div>"
)

if __name__ == "__main__":
    interface.launch()
