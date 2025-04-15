import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import random
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from data.dataset_c2d import MyDataset  # 自定义数据集类
from common_ultis.metrics import *
from common_ultis.utils import LinearDecayLR
from common_ultis.BaseShow import fnPlotTrainLine_WithField
import torch.nn as nn
from model.UNet_Down import GeneratorUNet


current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, feature1, feature2):
        similarity = self.cos_sim(feature1, feature2)
        loss = 1 - similarity.mean()
        return loss


def load_pretrained_down_blocks(model, pretrained_state_dict):
    model_state_dict = model.state_dict()
    pretrained_down_keys = [k for k in pretrained_state_dict.keys() if 'down' in k]
    pretrained_down_dict = {k: v for k, v in pretrained_state_dict.items() if k in pretrained_down_keys}
    model_state_dict.update(pretrained_down_dict)
    model.load_state_dict(model_state_dict)

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=512):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.layer4 = nn.Linear(hidden_size // 4, hidden_size // 8)
        self.layer5 = nn.Linear(hidden_size // 8, input_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, -1)  # 展平
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = x.view(b, c, h, w)
        return x

def main(seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs_list = [500, 400, 300, 200]
    batch_size = 64
    lr = 0.0001
    decay_start_epochs = [250, 200, 150, 100]
    set_seed(seed)


    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


    train_path_1 = '.../train'
    train_path_2 = '.../val'
    train_dataset_1 = MyDataset(path=train_path_1, input_transform=transform, target_transform=transform)
    train_dataset_2 = MyDataset(path=train_path_2, input_transform=transform, target_transform=transform)
    train_loader = DataLoader(train_dataset_1 + train_dataset_2 , batch_size=batch_size, shuffle=True, num_workers=8)

    E_CT = GeneratorUNet(1, 1).to(device)
    E_DWI = GeneratorUNet(1, 1).to(device)

    pretrained_ct_encoder_path = '.../weight/best_model.pth'
    pretrained_dwi_encoder_path = '.../weight/best_model.pth'


    pretrained_ct_dict = torch.load(pretrained_ct_encoder_path)
    load_pretrained_down_blocks(E_CT, pretrained_ct_dict)
    pretrained_dwi_dict = torch.load(pretrained_dwi_encoder_path)
    load_pretrained_down_blocks(E_DWI, pretrained_dwi_dict)


    sample_input = torch.randn(1, 1, 256, 256).to(device)
    with torch.no_grad():
        _, ct_features = E_CT(sample_input)
    feature_dims = [ct_features[i].view(1, -1).size(1) for i in range(len(ct_features))]

    # 定义保存路径
    save_path = '.../HFT_pretrain'
    os.makedirs(save_path, exist_ok=True)

    best_train_mae = float('inf')


    for param in E_CT.parameters():
        param.requires_grad = False
    for param in E_DWI.parameters():
        param.requires_grad = False


    for i in range(len(feature_dims)):
        input_dim = feature_dims[i]
        mlp = MLP(input_dim).to(device)

        criterion = CosineSimilarityLoss().to(device)
        optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=lr, betas=(0.5, 0.999))
        num_epochs = num_epochs_list[i]
        start_decay_epoch = decay_start_epochs[i]
        scheduler_mlp = LinearDecayLR(optimizer_mlp, start_epoch=start_decay_epoch, total_epochs=num_epochs)


        history = {
            "epoch": [],
            "train_loss": [],
            "train_mae": [],
        }

        for epoch in range(num_epochs):
            progress_description = f"Epoch [{epoch + 1}/{num_epochs}] - Scale {i + 1}/{len(feature_dims)}"
            train_bar = tqdm(train_loader, desc=progress_description)

            train_loss, train_mae = 0, 0

            E_CT.eval()
            E_DWI.eval()
            for _, (ct_images, dwi_images) in enumerate(train_bar):
                ct_images = ct_images.to(device)
                dwi_images = dwi_images.to(device)
                optimizer_mlp.zero_grad()


                with torch.no_grad():
                    _, ct_features = E_CT(ct_images)
                    _, dwi_features = E_DWI(dwi_images)

                ct_feature_flat = ct_features[i]
                dwi_feature_flat = dwi_features[i]

                trans_ct_feature = mlp(ct_feature_flat)

                loss = criterion(trans_ct_feature, dwi_feature_flat)
                loss.backward()
                optimizer_mlp.step()

                total_loss = loss.item()
                total_mae = calculate_mae(trans_ct_feature, dwi_feature_flat).item()

                train_loss += total_loss
                train_mae += total_mae

            train_loss /= len(train_loader)
            train_mae /= len(train_loader)


            scheduler_mlp.step()


            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_mae'].append(train_mae)

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Scale {i + 1}/{len(feature_dims)} "
                  f"Train Loss : {train_loss:.4f}, MAE: {train_mae:.4f}")

            if train_mae < best_train_mae:
                best_train_mae = train_mae
                torch.save(mlp.state_dict(), os.path.join(save_path, f'best_model_scale_{i + 1}.pth'))
                print(f"Saved best model for scale {i + 1} with MAE: {train_mae:.4f}")
            if (epoch + 1) % 100 == 0:
                torch.save(mlp.state_dict(), os.path.join(save_path, f'epoch{epoch + 1}_model_scale_{i + 1}.pth'))


        scale_save_path = os.path.join(save_path, f'scale_{i + 1}')
        os.makedirs(scale_save_path, exist_ok=True)
        fp_save_train = os.path.join(scale_save_path, 'pre_train.csv')
        pd.DataFrame(history).to_csv(fp_save_train, index=False)
        fnPlotTrainLine_WithField(
            inData=fp_save_train,
            inFields=["train_loss"],
            inYname="loss",
            inTitleName="loss",
            inDstDir=scale_save_path)
        fnPlotTrainLine_WithField(
            inData=fp_save_train,
            inFields=["train_mae"],
            inYname="MAE",
            inTitleName="MAE",
            inDstDir=scale_save_path)

if __name__ == "__main__":
    main(2)
