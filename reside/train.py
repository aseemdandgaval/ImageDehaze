import torch
from utils import (
    save_checkpoint,
    load_checkpoint,
    save_some_examples,
    calculate_psnr,
    save_psnr_to_csv,
    save_ssim_to_csv,
)
import torch.nn as nn
import torch.optim as optim
import config
from reside_dataset import RESIDEDataset

# from generator import Generator
from generator_upsample import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure as ssim

torch.backends.cudnn.benchmark = True

def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,  psnr_list, ssim_list
):
    loop = tqdm(loader, leave=True)
    psnr_batch_list = []
    ssim_batch_list = []

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        psnr_batch = calculate_psnr(y_fake, y).item()
        ssim_batch = ssim(y_fake, y).item()
        psnr_batch_list.append(psnr_batch)
        ssim_batch_list.append(ssim_batch)

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

    avg_psnr = sum(psnr_batch_list) / len(psnr_batch_list)
    avg_ssim = sum(ssim_batch_list) / len(ssim_batch_list)
    psnr_list.append(avg_psnr)
    ssim_list.append(avg_ssim)


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    # Loading the dataset and making the DataLoader
    dataset = RESIDEDataset(input_dir=config.RESIDE_INPUT_DIR, target_dir=config.RESIDE_TARGET_DIR)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(dataset=test_set, batch_size=config.BATCH_SIZE, shuffle=False)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    psnr_list = []
    ssim_list = []

    for epoch in range(config.NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch, config.NUM_EPOCHS))
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, psnr_list, ssim_list)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, dir=config.SAVE_IMAGES_DIR)

    save_psnr_to_csv(psnr_list, filename=config.SAVE_PSNR_CSV)
    save_ssim_to_csv(ssim_list, filename=config.SAVE_SSIM_CSV)


if __name__ == "__main__":
    main()
