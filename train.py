from ultralytics import YOLO #import YOLO algorithm from ultralyrics
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO("yolov8n.pt")  # build a pre-trained model from scratch
# Use the model
results = model.train(data="D:\YOLOv8 Local\Brain_Tumor_Detection\config.yaml", batch=60, epochs=50, name='brain_tumor_custom_#1')  # train the model