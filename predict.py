import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
from GAN import GAN, UNetGenerator  # 假设模型保存在 model.py 文件中

def main():
    parser = argparse.ArgumentParser(description='Generate image from GAN')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output', type=str, required=True, help='Path to save the generated image')
    parser.add_argument('--checkpoint', type=str, default='gan_training.ckpt', help='Path to the checkpoint file')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练模型
    model = GAN.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)

    # 加载图像
    input_image = Image.open(args.input).convert('RGB')
    original_size = input_image.size  # 记录原始尺寸

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整到固定大小，否则UNet处理会出问题
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device)  # 添加批处理维度

    # 推断
    with torch.no_grad():
        generated_image = model(input_tensor)

    # 后处理
    generated_image = generated_image.squeeze(0).cpu()  # 移除批处理维度并转到 CPU
    generated_image = generated_image * 0.5 + 0.5  # 反归一化
    generated_image = transforms.ToPILImage()(generated_image)

    # 调整输出图像大小到原始尺寸
    width, height = original_size
    generated_image = transforms.Resize((height, width))(generated_image)

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generated_image.save(args.output)
    print(f"Generated image saved to {args.output} (original size maintained)")

if __name__ == "__main__":
    main()