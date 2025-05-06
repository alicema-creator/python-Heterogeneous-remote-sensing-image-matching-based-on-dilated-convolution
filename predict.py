import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# 主函数
def main():
    # 加载模型
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return
    
    print("加载模型中...")
    model = load_model(model_path)
    print("模型加载完成")
    
    # 预测val文件夹
    val_folder = "val"
    if not os.path.exists(val_folder):
        print(f"错误: 验证文件夹 {val_folder} 不存在")
        return
    
    print(f"\n开始处理 {val_folder} 文件夹中的图像...")
    predict_on_val_folder(model, val_folder)

if __name__ == "__main__":
    main()