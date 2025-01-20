import os
import torch
import config
import pandas as pd
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio
from torchvision.utils import save_image

def save_some_examples(gen, val_loader, epoch, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, dir + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, dir + f"/input_{epoch}.png")
        # if epoch == 1:
        save_image(y * 0.5 + 0.5, dir + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def calc_psnr_per_batch(generated_batch, label_batch):
    generated_batch = generated_batch.detach().cpu().numpy()
    label_batch = label_batch.detach().cpu().numpy()

    psnr = []
    for i in range(generated_batch.shape[0]):
        psnr.append(peak_signal_noise_ratio(generated_batch[i], label_batch[i]))

    return sum(psnr) / len(psnr)


def calculate_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return psnr

def save_psnr_to_csv(psnr_list, filename="psnr_log.csv"):
    df = pd.DataFrame(psnr_list, columns=["PSNR"])
    df.to_csv(filename, index=False)

def save_ssim_to_csv(ssim_list, filename="ssim_log.csv"):
    df = pd.DataFrame(ssim_list, columns=["SSIM"])
    df.to_csv(filename, index=False)
