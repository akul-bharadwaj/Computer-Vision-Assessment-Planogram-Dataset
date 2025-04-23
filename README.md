# Computer-Vision-Assessment_Planogram-Dataset

## Project Overview
This project is for a Gradio based application that analyzes retail shelf images and scores them based on compliance.

## Gradio App
You can access the Gradio application for analyzing retail shelf gaps and compliance at the following link:

[Retail Shelf Gap and Compliance Analyzer](https://huggingface.co/spaces/akul-29/Retail-Shelf-Gap-and-Compliance-Analyzer)

## Fine-tuned YOLOv8 model for retail shelf gaps
You can find the fine-tuned YOLOv8 model for detecting retail shelf gaps at the following link:

[Retail Shelf Gap Detection Model](https://huggingface.co/akul-29/Retail-Shelf-Gap-Detection_Model)

## Repository Structure
```
├── datasets/shelf_planograms/DATASET_Planogram
│   ├── train/
│   │   ├── images/            # Training images for model learning
│   │   ├── labels/            # Corresponding YOLO-format labels for training
│   ├── val/
│   │   ├── images/            # Validation images used during training
│   │   ├── labels/            # YOLO-format labels for validation set
│   ├── test/
│   │   ├── images/            # Test images to evaluate final model
│   │   ├── labels/            # Ground truth labels for test images (if available)
│   ├── data.yaml              # Dataset configuration file for YOLOv5
│
├── runs/detect/train/
│   ├── weights/
│   │   ├── best.pt            # Best-performing model weights (used for inference)
│
├── gap_detection_model.ipynb  # Jupyter notebook used for model development and experimentation
├── app.py                     # Gradio-based interactive application script
├── output_YOLO_1.jpg          # Sample annotated result (YOLO output image)
├── output_YOLO_2.jpg          # Additional sample output image
├── requirements.txt           # Required Python packages and versions
├── README.md                  # Project documentation with usage and setup instructions

```

---
