import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os



# 5. 主函数
def main():
    # 加载模型
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return
    
    print("加载模型中...")
    model = load_model(model_path)
    print("模型加载完成")
    
    # 匹配图像
    image_dir = "images/6"
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录 {image_dir} 不存在")
        return
    
    print(f"\n对 {image_dir} 中的图像进行特征匹配...")
    match_images(model, image_dir)

if __name__ == "__main__":
    main()