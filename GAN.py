import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

import os
from PIL import Image
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset

class PairedDataset(Dataset):
    def __init__(self, input_root, target_root, transform=None):
        # 扩展用户路径（处理 ~）
        self.input_root = os.path.expanduser(input_root)
        self.target_root = os.path.expanduser(target_root)

        self.input_files = sorted(os.listdir(self.input_root))
        self.target_files = sorted(os.listdir(self.target_root))
        self.transform = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_img_path = os.path.join(self.input_root, self.input_files[idx])
        target_img_path = os.path.join(self.target_root, self.target_files[idx])
        input_img = Image.open(input_img_path).convert('RGB')
        target_img = Image.open(target_img_path).convert('RGB')

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img

# UNet Generator
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VGGDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(VGGDiscriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=3, padding=1),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)  # 从 [Batch, 1, H, W] -> [Batch, 1, 1, 1]
        x = self.sigmoid(x)
        return x.view(x.size(0), -1)  # 转换为 [Batch, 1]



# GAN Model
class GAN(pl.LightningModule):
    def __init__(self, lr=0.0002, b1=0.5, b2=0.999):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # 手动管理优化器

        # Networks
        self.generator = UNetGenerator()
        self.discriminator = VGGDiscriminator()

        self.adversarial_loss = nn.BCELoss()

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x 是输入图像（带高光），y 是标签图像（无高光）

        # 前向传播生成器
        self.generated_imgs = self(x)
        # 获取优化器
        opt_g, opt_d = self.optimizers()

        # 训练生成器
        opt_g.zero_grad()
        valid = torch.ones(y.size(0), 1).type_as(y)  # 真实标签
        # 对抗损失（BCE 损失）
        adversarial_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
        # 像素级 L1 损失
        l1_loss = F.l1_loss(self.generated_imgs, y)
        # 总生成器损失 = 对抗损失 + 像素级损失
        g_loss = adversarial_loss + 100 * l1_loss  # 100 是超参数，可根据需要调整
        
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        opt_g.step()

        # 训练鉴别器
        opt_d.zero_grad()
        # 真实图像的损失
        real_out = self.discriminator(y)
        real_loss = self.adversarial_loss(real_out, torch.ones_like(real_out))
        # 生成图像的损失
        fake_out = self.discriminator(self.generated_imgs.detach())
        fake_loss = self.adversarial_loss(fake_out, torch.zeros_like(fake_out))

        d_loss = (real_loss + fake_loss) / 2

        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()

        return {"loss": d_loss}  # 返回一个字典或 None

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = PairedDataset(
            input_root="~/datasets/PSD/PSD_Train/spec",  # 输入图像路径
            target_root="~/datasets/PSD/PSD_Train/nospec",  # 标签图像路径
            transform=transform
        )

        return DataLoader(dataset, batch_size=2, shuffle=True)


# Train the model
if __name__ == "__main__":
    import argparse, glob, os, torch
    from pytorch_lightning.loggers import TensorBoardLogger

    torch.set_float32_matmul_precision('medium')
    VERSION = 'V1'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='从checkpoint恢复训练的路径')
    args = parser.parse_args()

    logger = TensorBoardLogger(save_dir='./', name='lightning_logs', version=VERSION)

    # 判断是否存在可用checkpoint
    ckpt_path = args.resume or max(glob.glob(os.path.join('lightning_logs', VERSION, 'checkpoints', '*.ckpt')), 
                                    key=os.path.getmtime, default=None)
    print(f"自动恢复训练: {ckpt_path}" if ckpt_path else "开始新训练")

    trainer = pl.Trainer(
        accelerator='gpu', devices='auto', max_epochs=25,
        logger=logger, enable_checkpointing=True, default_root_dir='.',
        num_sanity_val_steps=0,
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join('lightning_logs', VERSION, 'checkpoints'),
            filename='{epoch:02d}-{step}',
            save_last=True,
            every_n_epochs=5  # 每5个epoch保存一次模型
        )]
    )

    model = GAN()
    # 直接调用 trainer.fit(model, train_loader)
    trainer.fit(model, ckpt_path=ckpt_path)