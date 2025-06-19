import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('runs/train/1/yolo11-ContextGuidedDown/weights/best.pt') # select your model.pt path
    model.predict(source='datasets/images/val',
                  imgsz=640,
                  project='runs/detect/1',
                  name='yolo11-ContextGuidedDown1',
                  save=True,
                )