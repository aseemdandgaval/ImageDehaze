import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESIDE_INPUT_DIR = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/RESIDE/hazy"
RESIDE_TARGET_DIR = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/RESIDE/clear"

LEARNING_RATE = 2e-4
BATCH_SIZE = 32
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 10e-5
LAMBDA_GP = 10
NUM_EPOCHS = 1

LOAD_MODEL = True
SAVE_MODEL = False

CHECKPOINT_DISC = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/reside/checkpoints/disc.pth.tar"
CHECKPOINT_GEN = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/reside/checkpoints/gen.pth.tar"

CHECKPOINT_DISC_UPSAMPLE = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/reside/checkpoints_upsample/disc_upsample.pth.tar"
CHECKPOINT_GEN_UPSAMPLE = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/reside/checkpoints_upsample/gen_upsample.pth.tar"

SAVE_IMAGES_DIR = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/reside/evaluation"
SAVE_IMAGES_DIR_UPSAMPLE = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/reside/evaluation_upsample"

SAVE_PSNR_CSV = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/reside/logs/psnr_reside_log.csv"
SAVE_PSNR_CSV_UPSAMPLE = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/reside/logs/psnr_reside_umpsample_log.csv"

SAVE_SSIM_CSV = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/reside/logs/ssim_reside_log.csv"
SAVE_SSIM_CSV_UPSAMPLE = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/reside/logs/ssim_reside_umpsample_log.csv"