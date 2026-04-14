from ultralytics import YOLO
import torch

device = "mps" 

name = "2026_04_02(2)"  # 模型路徑

model_path = f'yolo_output/runs/detect/{name}/weights/best.pt'
# 驗證集資料夾（需有標註檔案如YOLO格式的labels）
data_yaml = 'yolo_data/combined_with_synthesis_od_v5/data.yaml'

model = YOLO(model_path)
model.to(device)


# 評估模型
classes = [0]  # 只評估結節類別（0）
metrics = model.val(data=data_yaml, conf=0.2, iou=0.5, name = f"{name}_val_results", classes=classes)
print(metrics.results_dict)