import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('ultralytics-yolo11-main/ultralytics/cfg/models/11/yolo11-ContextGuidedDown.yaml')
    model.train(data='ultralytics-yolo11-main/datasets/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=64,
                close_mosaic=0, # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
                workers=8, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                optimizer='SGD', # using SGD
                project='runs/train/1',
                name='yolo11-ContextGuidedDown',
                )