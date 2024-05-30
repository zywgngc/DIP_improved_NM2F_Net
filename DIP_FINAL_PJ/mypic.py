import torch
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from model_backup import DM2FNet  # 确保引入了正确的模型类

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 载入模型
model_path = '/root/dip/DM2F-Net-master/ckpt/O-Haze/iter_40000_loss_0.01462_lr_0.000000.pth'
model = DM2FNet().cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

# 图片处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 输出目录
output_dir = './dehazed_images'
os.makedirs(output_dir, exist_ok=True)

# 处理并保存去雾后的图片
for i in range(1, 6):
    image_path = f'/root/dip/DM2F-Net-master/data/MY5/{i}.png'
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        dehazed_tensor = model(input_tensor)

    # 保存去雾后的图片
    save_image(dehazed_tensor, os.path.join(output_dir, f'dehazed_{i}.png'))

print("Dehazing completed. Results are saved in", output_dir)