import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)
import random
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from data.dataset_pretrain import MyDataset  # 自定义数据集类
from model.UNet import *
from common_ultis.metrics import *
from common_ultis.utils import LinearDecayLR
from common_ultis.BaseShow import fnPlotTrainLine_WithField
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多卡训练
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 200
    batch_size = 16
    lr = 0.001
    start_decay_epoch = num_epochs/2
    set_seed(seed)


    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


    train_ct_path= '.../train'
    train_dataset = MyDataset(path=train_ct_path, input_transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)


    generator = GeneratorUNet(1, 1).to(device)


    criterion_L2 = nn.MSELoss().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    scheduler_G = LinearDecayLR(optimizer_G, start_epoch=start_decay_epoch, total_epochs=num_epochs)



    history = {
        "epoch": [],
        "train_loss": [],
        "train_ssim": [],
        "train_mae": [],
    }


    save_path = '.../SSL_pretrain'
    os.makedirs(save_path, exist_ok=True)

    best_train_mae = float('inf')

    # 训练循环
    for epoch in range(num_epochs):
        progress_description = f"Epoch [{epoch + 1}/{num_epochs}]"
        train_bar = tqdm(train_loader, desc=progress_description)

        train_loss, train_ssim, train_mae = 0, 0, 0
        generator.train()
        for _, (mask_images, images) in enumerate(train_bar):
            mask_images = mask_images.to(device)
            images = images.to(device)
            optimizer_G.zero_grad()
            rec_images = generator(mask_images)
            loss = criterion_L2(rec_images, images)
            loss.backward()
            optimizer_G.step()

            train_loss += loss.item()
            train_ssim += calculate_ssim_single_channel(rec_images, images)
            train_mae += calculate_mae(rec_images, images)

        train_loss /= len(train_loader)
        train_ssim /= len(train_loader)
        train_mae /= len(train_loader)



        scheduler_G.step()


        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_ssim'].append(train_ssim)
        history['train_mae'].append(train_mae)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss : {train_loss:.4f}, SSIM: {train_ssim:.4f}, MAE: {train_mae:.4f} ")

        if train_mae < best_train_mae:
            best_train_mae = train_mae
            torch.save(generator.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"Saved best model with MAE: {train_mae:.4f}")
        if (epoch + 1) % 100 == 0:
                torch.save(generator.state_dict(), os.path.join(save_path, f'pre_model{epoch + 1}.pth'))


    fp_save_train = os.path.join(save_path, 'pre_train.csv')
    pd.DataFrame(history).to_csv(fp_save_train, index=False)
    fnPlotTrainLine_WithField(
        inData=fp_save_train,
        inFields=["train_loss"],
        inYname="loss",
        inTitleName="loss",
        inDstDir=save_path)
    fnPlotTrainLine_WithField(
        inData=fp_save_train,
        inFields=["train_mae"],
        inYname="MAE",
        inTitleName="MAE",
        inDstDir=save_path)
    fnPlotTrainLine_WithField(
        inData=fp_save_train,
        inFields=["train_ssim"],
        inYname="ssim",
        inTitleName="ssim",
        inDstDir=save_path)



if __name__ == "__main__":
    main(2)
