from ultralytics import YOLO
import os, json

img_dir = r"E:\Documents\WashU\Senior\Second Semester\Project\sapiens\lite\data\downtown_cafe_test"
out_json = r"E:\tmp\person_bboxes.json"

model = YOLO("yolov8n.pt")
results = model.predict(source=img_dir, conf=0.25, classes=[0], verbose=False)  # person=0

all_boxes = {}
for r in results:
    boxes = []
    if r.boxes is not None:
        for b in r.boxes.xyxy.cpu().numpy():
            x1,y1,x2,y2 = b.tolist()
            boxes.append([x1,y1,x2,y2])
    all_boxes[os.path.basename(r.path)] = boxes

with open(out_json, "w") as f:
    json.dump(all_boxes, f)
print("saved:", out_json)
