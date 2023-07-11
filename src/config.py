import torch

# HYPERPARAMETERS
BATCH_SIZE = 2  # increase / decrease according to GPU memeory
RESIZE_TO = 512  # resize the image for training and transforms
NUM_EPOCHS = 10  # number of epochs to train for
DETECTION_THRESHOLD = 0.9  # threshold for predicting classes in images

# DEVICE AND DIRECTORIES
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TRAIN_DIR = '../data/train'  # training images and XML files directory
VALID_DIR = '../data/validate'  # validation images and XML files directory
TEST_DIR = '../data/test' # test images directory
OUT_DIR = '../outputs'  # location to save model and plots


# CLASSES (0 index is reserved for background)
CLASSES = ['background', 'button']
NUM_CLASSES = len(CLASSES)

# MANUALS
VISUALIZE_TRANSFORMED_IMAGES = False  # whether to visualize images after creating the data loaders
SAVE_PLOTS_EPOCH = 1  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 1  # save model after these many epochs

