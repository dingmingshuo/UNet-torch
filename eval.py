from model.unet import UNet
from model.loss import CrossEntropyWithLogits
from data.dataloader import create_val_dataset
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np


data_dir = "/home/mingshuo/Dataset/ISBI"
model_dir = "/home/mingshuo/Workspace/UNet-torch/result/step=all"

val_dataloader = create_val_dataset(data_dir)
model = UNet(1, 2)
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint)
model.eval()

t_rescale_image = transforms.Normalize(mean=(-1), std=(2))

with torch.no_grad():
    for img, mask in val_dataloader:
        # print(img.shape)
        # img = t_rescale_image(img).squeeze().numpy()
        # img = Image.fromarray(img, mode='L')
        # img.show()
        # print(img)
        pred = model(img)
        pred = torch.nn.Softmax(dim=1)(pred)[:, 1, :, :].squeeze().numpy()
        pred = pred*255
        pred = pred.astype(np.uint8)
        pred = Image.fromarray(pred, mode='L')
        pred.show()
        break
