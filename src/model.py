import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



def create_model(num_classes):
    # load Faster RCNN pre-trained model
    #os.environ['TORCH_HOME'] = '/Users/paulafrindte/Library/CloudStorage/OneDrive-TUM/Dokumente/4_Uni/2323SS/TechChallenge/FiberFlashPytorch/src'  # setting the environment variable
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model