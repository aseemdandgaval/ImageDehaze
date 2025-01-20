import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NHHAZE_ROOT_DIR = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/NH-HAZE"

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 100

LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_DISC = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/nhhaze/checkpoints/disc.pth.tar"
CHECKPOINT_GEN = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/nhhaze/checkpoints/gen.pth.tar"

CHECKPOINT_DISC_UPSAMPLE = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/nhhaze/checkpoints_upsample/disc_upsample.pth.tar"
CHECKPOINT_GEN_UPSAMPLE = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/nhhaze/checkpoints_upsample/gen_upsample.pth.tar"


SAVE_IMAGES_DIR = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/nhhaze/evaluation"
SAVE_IMAGES_DIR_UPSAMPLE = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/nhhaze/evaluation_upsample"

SAVE_PSNR_CSV = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/nhhaze/logs/psnr_nhhaze_log.csv"
SAVE_PSNR_CSV_UPSAMPLE = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/ECE285-Project/pix2pix/nhhaze/logs/psnr_nhhaze_umpsample_log.csv"