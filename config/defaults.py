from yacs.config import CfgNode as CN

# ----------------------------------------------------------------
# Config definition
# ----------------------------------------------------------------

_C = CN()

# ----------------------------------------------------------------
# Training Parameters
# ----------------------------------------------------------------
# path of cover images
_C.COVER_PATH = "../data/cover"
# path of secret images (images to be hidden inside cover images)
_C.SECRET_PATH = "../data/secret"
# path to store checkpoints
_C.CHECKPOINT_PATH = "./checkpoints"
# learninig rate of Embedder
_C.EM_LR = 0.001
# learninig rate of Extractor
_C.EX_LR = 0.001
# learninig rate of already trained Steganalyzer
_C.ST_LR = 0.001
# learninig rate of discriminator of extractor
_C.DI_LR = 0.001
# No. of epochs to train
_C.NUM_EPOCHS = 400
# Training batch size
_C.BATCH_SIZE = 16
# No. of training images to be used
_C.NUM_IMAGES = 50000
# Size of each training images
_C.IMAGE_SIZE = (256, 256)
# No. of channels in input image (RGB=3, Gray=1)
_C.NUM_CHANNELS = 3
# Weightage of Embedder MSE loss
_C.WT_MSE = 1.0
# Weightage of Adversarial loss coming from Steganalyzer
_C.WT_ADV = 0.2
# Weightage of Extractor MSE loss
_C.WT_EXT = 0.3
# Weightage of Adversarial loss coming from discriminator of extractor
_C.WT_EXT_ADV = 0.1
# Weightage of Perceptual loss in Extractor
_C.WT_PERCEPT = 0.0004