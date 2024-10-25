from ultralytics import YOLO

model = YOLO("/home/Niwenjin/Proj/mcrs/models/yolov8n_flag_0926.pt")

results = model.train(
    project="runs/flag",
    data="flag.yaml",
    epochs=100,
    batch=64,
    # optimizer="SGD",
    # lr0=1e-3,
    # warmup_epochs=0,  # 微调时设为0
    box=5.0,  # (float) box loss gain
    cls=0.5,  # (float) cls loss gain (scale with pixels)
    dfl=4.0,  # (float) dfl loss gain
    freeze=22,  # 冻结层数
    # neg_dir="/home/Niwenjin/Proj/mcrs/dataset/flag/data/neg/",  # 负样本文件夹
    # neg_num=2,  # 负样加入数
)

# model = YOLO("yolov8n.pt")

# results = model.train(
#     project="runs/body",
#     data="porn.yaml",
#     epochs=100,
#     batch=16,
#     # optimizer="SGD",
#     # lr0=1e-3,
#     # warmup_epochs=0,  # 微调时设为0
#     freeze=10,
# )
