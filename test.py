import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from data.dataset_c2d import MyDataset
from common_ultis.metrics import calculate_metrics
import re
from model.EHTGAN import *
import torchvision.transforms.functional as F

def main(seed, mode):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    test_path ='.../test'


    test_dataset = MyDataset(path=test_path, input_transform=transform, target_transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)


    generator = GeneratorUNet().to(device)
    generator.load_state_dict(torch.load('.../weight/best_model.pth'))


    save_dir = '../EHTGAN'
    os.makedirs(save_dir, exist_ok=True)
    csv_filename = os.path.join(save_dir, '{}.csv'.format(mode))
    csv_columns = ['Image', 'SSIM', "MAE", "PSNR", "MAPE"]
    test_results = []


    image_save_dir = save_dir + '/{}'.format(mode)
    os.makedirs(image_save_dir, exist_ok=True)


    def extract_image_id(filename):
        match = re.findall(r'\d+', filename)
        if len(match) >= 2:
            return f"{match[-2]}_{match[-1]}"
        return None

    generator.eval()

    with torch.no_grad():
        total_ssim = 0
        total_mae = 0
        total_psnr = 0
        total_mape = 0
        num_images = len(test_loader)
        print(num_images)
        for i, (real_images, target_images) in enumerate(test_loader):
            real_images = real_images.to(device)
            target_images = target_images.to(device)


            fake_images, _, _ = generator(real_images)


            ssim_value, mae_value, psnr_value, mape_value = calculate_metrics(target_images, fake_images)
            total_ssim += ssim_value
            total_mae += mae_value
            total_psnr += psnr_value
            total_mape += mape_value


            image_id = test_dataset.image_paths[i]


            test_results.append({'Image': image_id, 'SSIM': ssim_value.item(), "MAE": mae_value.item(), "PSNR": psnr_value.item(), "MAPE": mape_value.item()})


            real_images = real_images.to(fake_images.dtype)


            num_channels = fake_images.size(1)


            real_images_broadcasted = real_images.expand(-1, num_channels, -1, -1)


            combined_image = torch.cat((real_images_broadcasted, target_images, fake_images), dim=-1)
            combined_image = combined_image.squeeze(0).cpu()


            combined_image = combined_image * 0.5 + 0.5


            save_image(combined_image, os.path.join(image_save_dir, f'{image_id}'))


    average_ssim = total_ssim / num_images
    average_mae = total_mae / num_images
    average_psnr = total_psnr / num_images
    average_mape = total_mape / num_images


    df = pd.DataFrame(test_results, columns=csv_columns)
    df.to_csv(csv_filename, index=False)

    print(f"Test results saved to {csv_filename}")
    print(f"Average SSIM: {average_ssim:.3f}")
    print(f"Average MAE: {average_mae:.3f}")
    print(f"Average PSNR: {average_psnr:.3f}")
    print(f"Average MAPE: {average_mape:.3f}")


if __name__ == "__main__":
    main(2, 'test')
