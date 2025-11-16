from ultralytics import YOLO
# engineresults
# if pred_boxes and show_boxes:

def predictmodel(predictConfig,sourceConfig):
      # 开始加载模型
      model = YOLO(predictConfig)
      # 指定训练参数开始测试
      for i in model.predict(source=sourceConfig, stream=True, conf=0.15, iou=0.55,
                             project="runs/predict", name='exp', save_txt=True, save=True):
            print(i)

if __name__ == "__main__":
      # 填写测试的网络模型名称
      predictConfig = "runs/train/exp/weights/best.pt"

      # 填写测试图片文件夹
      sourceConfig = 'datasets/train_shoes/val/images'

      # 调用测试方法
      predictmodel(predictConfig,sourceConfig)

























































