from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n_flag_0925.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model

results = model.train(
    data="flag.yaml",
    epochs=200,
    batch=128,
    optimizer="AdamW",
    imgsz=640,
    device="0,1,2,3",
    # mixup=1.0,
    # copy_paste=1.0,
)
