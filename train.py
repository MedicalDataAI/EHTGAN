import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from data.dataset_c2d import MyDataset  # 自定义数据集类
from model.EHTGAN import *
from common_ultis.metrics import *
from common_ultis.utils import LinearDecayLR
from common_ultis.BaseShow import fnPlotTrainLine_WithField


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_pretrained_up_blocks(model, pretrained_state_dict):
    model_state_dict = model.state_dict()
    pretrained_up_keys = [k for k in pretrained_state_dict.keys() if 'up' in k]
    pretrained_up_dict = {k: v for k, v in pretrained_state_dict.items() if k in pretrained_up_keys}
    model_state_dict.update(pretrained_up_dict)
    model.load_state_dict(model_state_dict)
def load_pretrained_down_blocks(model, pretrained_state_dict):
    model_state_dict = model.state_dict()
    pretrained_down_keys = [k for k in pretrained_state_dict.keys() if 'down' in k]
    pretrained_down_dict = {k: v for k, v in pretrained_state_dict.items() if k in pretrained_down_keys}
    model_state_dict.update(pretrained_down_dict)
    model.load_state_dict(model_state_dict)
def load_hft_pretrained(model, pretrained_paths):
    mlp_layers = [model.mlp1, model.mlp2, model.mlp3, model.mlp4]
    for mlp, path in zip(mlp_layers, pretrained_paths):
        pretrained_dict = torch.load(path)
        mlp_dict = mlp.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in mlp_dict and mlp_dict[k].shape == v.shape}
        mlp_dict.update(pretrained_dict)
        mlp.load_state_dict(mlp_dict)
def load_lft_pretrained(model, pretrained_paths):
    cnn_layers = [model.cnn1, model.cnn2, model.cnn3, model.cnn4]
    for cnn, path in zip(cnn_layers, pretrained_paths):
        pretrained_dict = torch.load(path)
        cnn_dict = cnn.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in cnn_dict and cnn_dict[k].shape == v.shape}
        cnn_dict.update(pretrained_dict)
        cnn.load_state_dict(cnn_dict)

def main(seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 200
    batch_size = 16
    lr = 0.0002
    lambda_L1 = 100
    start_decay_epoch = 100
    feature_weights = [0.6] * 4 + [0.4] * 4
    set_seed(seed)

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    train_data_path = '.../train'
    train_dataset = MyDataset(path=train_data_path, input_transform=transform, target_transform=transform, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    val_data_path = '.../val'
    val_dataset = MyDataset(path=val_data_path, input_transform=transform, target_transform=transform, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


    pretrained_encoder_path = '.../weight/best_model.pth'
    pretrained_decoder_path = '.../weight/best_model.pth'
    pretrained_hft_paths = [
        ".../weight/model_scale_4.pth",
        "../weight/model_scale_3.pth",
        "../weight/model_scale_2.pth",
        "../weight/model_scale_1.pth"
    ]

    pretrained_lft_paths = [
        "../weight/model_scale_1.pth",
        "../weight/model_scale_2.pth",
        "../weight/model_scale_3.pth",
        "../weight/model_scale_4.pth"
    ]

    # 定义生成器
    generator = GeneratorUNet().to(device)

    # 加载预训练模型权重
    pretrained_down_dict = torch.load(pretrained_encoder_path)
    load_pretrained_down_blocks(generator, pretrained_down_dict)

    pretrained_up_dict = torch.load(pretrained_decoder_path)
    load_pretrained_up_blocks(generator, pretrained_up_dict)

    load_hft_pretrained(generator, pretrained_hft_paths)
    load_lft_pretrained(generator, pretrained_lft_paths)


    generator.freeze_cm()
    generator.eval_cm()

    # 获得不同尺度特征标签
    encoder = GeneratorUNet().to(device)
    pretrained_down_dict_dict = torch.load(pretrained_decoder_path)
    load_pretrained_down_blocks(encoder, pretrained_down_dict_dict)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    discriminator = CBAMDiscriminator(1).to(device)
    discriminator.apply(weights_init_normal)


    criterion_feature = CosineSimilarityLoss().to(device)
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    criterion_L1 = nn.L1Loss().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


    scheduler_G = LinearDecayLR(optimizer_G, start_epoch=start_decay_epoch, total_epochs=num_epochs)
    scheduler_D = LinearDecayLR(optimizer_D, start_epoch=start_decay_epoch, total_epochs=num_epochs)


    history = {
        "epoch": [],
        "train_loss_G": [],
        "train_loss_D": [],
        "train_ssim": [],
        "train_mae": [],
        "val_loss_G": [],
        "val_loss_D": [],
        "val_ssim": [],
        "val_mae": []
    }

    # 定义保存路径
    save_path = '.../EHTGAN'
    os.makedirs(save_path, exist_ok=True)
    # 初始化最小验证集MAE
    best_val_mae = float('inf')
    best_val_ssim = 0
    # 训练循环
    for epoch in range(num_epochs):
        progress_description = f"Epoch [{epoch + 1}/{num_epochs}]"
        train_bar = tqdm(train_loader, desc=progress_description)

        train_loss_G, train_loss_D = 0, 0
        train_ssim, train_mae = 0, 0

        for _, (real_images, target_images) in enumerate(train_bar):
            real_images = real_images.to(device)
            target_images = target_images.to(device)


            optimizer_G.zero_grad()

            fake_images, _, fake_features = generator(real_images)

            with torch.no_grad():
                _, target_features, _ = encoder(target_images)

            fake_output = discriminator(fake_images, real_images)
            loss_G_GAN = criterion_GAN(fake_output, torch.ones_like(fake_output, requires_grad=False))


            loss_G_F = sum([w * criterion_feature(fake, real) for fake, real, w in
                            zip(fake_features, target_features, feature_weights)])

            loss_G_L1 = criterion_L1(fake_images, target_images) * lambda_L1

            loss_G = loss_G_GAN + loss_G_L1 + loss_G_F

            loss_G.backward()
            optimizer_G.step()


            optimizer_D.zero_grad()

            real_output = discriminator(target_images, real_images)
            loss_D_real = criterion_GAN(real_output, torch.ones_like(real_output)-0.1)

            fake_output = discriminator(fake_images.detach(), real_images)
            loss_D_fake = criterion_GAN(fake_output, torch.zeros_like(fake_output)+0.1)

            loss_D = (loss_D_real + loss_D_fake) * 0.5

            loss_D.backward()
            optimizer_D.step()

            train_loss_G += loss_G.item()
            train_loss_D += loss_D.item()
            train_ssim += calculate_ssim_single_channel(fake_images, target_images).item()
            train_mae += calculate_mae(fake_images, target_images).item()

        train_loss_G /= len(train_loader)
        train_loss_D /= len(train_loader)
        train_ssim /= len(train_loader)
        train_mae /= len(train_loader)

        # 验证
        val_loss_G, val_loss_D = 0, 0
        val_ssim, val_mae = 0, 0  # 添加MAE计算

        with torch.no_grad():
            for real_images, target_images in val_loader:
                real_images = real_images.to(device)
                target_images = target_images.to(device)

                fake_images, _, fake_features = generator(real_images)

                _, target_features, _ = encoder(target_images)

                real_output = discriminator(target_images, real_images)
                fake_output = discriminator(fake_images.detach(), real_images)

                loss_D_real = criterion_GAN(real_output, torch.ones_like(real_output)-0.1)
                loss_D_fake = criterion_GAN(fake_output, torch.zeros_like(fake_output)+0.1)
                loss_D = (loss_D_real + loss_D_fake) * 0.5

                loss_G_GAN = criterion_GAN(fake_output, torch.ones_like(fake_output))
                loss_G_L1 = criterion_L1(fake_images, target_images) * lambda_L1

                # 特征损失
                loss_G_F = sum([w * criterion_feature(fake, real) for fake, real, w in
                                zip(fake_features, target_features, feature_weights)])

                loss_G = loss_G_GAN + loss_G_L1 + loss_G_F

                val_loss_G += loss_G.item()
                val_loss_D += loss_D.item()
                val_ssim += calculate_ssim_single_channel(fake_images, target_images).item()
                val_mae += calculate_mae(fake_images, target_images).item()

        val_loss_G /= len(val_loader)
        val_loss_D /= len(val_loader)
        val_ssim /= len(val_loader)
        val_mae /= len(val_loader)


        scheduler_G.step()
        scheduler_D.step()

        # 打印和记录
        history['epoch'].append(epoch + 1)
        history['train_loss_G'].append(train_loss_G)
        history['train_loss_D'].append(train_loss_D)
        history['train_ssim'].append(train_ssim)
        history['train_mae'].append(train_mae)
        history['val_loss_G'].append(val_loss_G)
        history['val_loss_D'].append(val_loss_D)
        history['val_ssim'].append(val_ssim)
        history['val_mae'].append(val_mae)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss G: {train_loss_G:.4f}, D: {train_loss_D:.4f}, SSIM: {train_ssim:.4f}, MAE: {train_mae:.4f} "
              f"Val Loss G: {val_loss_G:.4f}, D: {val_loss_D:.4f}, SSIM: {val_ssim:.4f}, MAE: {val_mae:.4f}")


        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            torch.save(generator.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"Saved best model with SSIM: {val_ssim:.4f}")


        if (epoch + 1) % 100 == 0:
            torch.save(generator.state_dict(), os.path.join(save_path, f'e_{epoch + 1}.pth'))

    # 保存训练历史
    fp_save_train = os.path.join(save_path, 'train.csv')
    pd.DataFrame(history).to_csv(fp_save_train, index=False)
    fnPlotTrainLine_WithField(
        inData=fp_save_train,
        inFields=["train_loss_G", "val_loss_G"],
        inYname="loss",
        inTitleName="loss_G",
        inDstDir=save_path)
    fnPlotTrainLine_WithField(
        inData=fp_save_train,
        inFields=["train_loss_D", "val_loss_D"],
        inYname="loss",
        inTitleName="loss_D",
        inDstDir=save_path)
    fnPlotTrainLine_WithField(
        inData=fp_save_train,
        inFields=["train_ssim", "val_ssim"],
        inYname="ssim",
        inTitleName="ssim",
        inDstDir=save_path)
    fnPlotTrainLine_WithField(
        inData=fp_save_train,
        inFields=["train_mae", "val_mae"],
        inYname="MAE",
        inTitleName="MAE",
        inDstDir=save_path)


if __name__ == "__main__":
    main(seed=2)
