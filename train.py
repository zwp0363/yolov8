from ultralytics import YOLO
import warnings
import argparse
warnings.filterwarnings('ignore')

"""
 可以直接将该cfg路径替换到下面 切换不同改进的模型网络 # 填写训练的网络模型名称
 
 ultralytics/cfg/models/mycfg/yolov8n.yaml
 
 ultralytics/cfg/models/mycfg/yolov8n-2-SPPF_LSKA.yaml

 ultralytics/cfg/models/mycfg/yolov8n-3-VoVGSCSP.yaml

 ultralytics/cfg/models/mycfg/yolov8n-4-VoVGSCSP-SPPF_LSKA.yaml

"""
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights',type=str, default='yolov8n.pt', help='loading pretrain weights') # 预训练权重
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/mycfg/yolov8n.yaml', help='models') # 填写训练的网络模型名称
    parser.add_argument('--data', type=str, default='datasets.yaml', help='datasets') # 数据集配置文件
    parser.add_argument('--epochs', type=int, default=100, help='train epoch') # 模型训练轮次
    parser.add_argument('--batch', type=int, default=4, help='total batch size for all GPUs') # 批次大小
    parser.add_argument('--imgsz', type=int, default=640, help='image sizes') # 默认输入大小 640*640
    parser.add_argument('--optimizer', default='SGD', help='use optimizer') # 优化器
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers') #多线程数量
    parser.add_argument('--project', default='runs/train', help='save to project/name') #文件夹结果
    parser.add_argument('--name', default='exp', help='save to project/name') #文件夹结果
    return parser.parse_args()

if __name__ == '__main__':
    args=main()
    # 开始加载模型
    model = YOLO(args.cfg)
    if(".pt" in args.weights):
        pass
        print("+++++++载入预训练权重：",args.weights,"++++++++")
        model.load(args.weights)
    else:
        pass
        print("-------没有载入预训练权重-------")
    # 指定训练参数开始训练
    model.train(data=args.data,
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                optimizer=args.optimizer,
                project=args.project,
                name=args.name)