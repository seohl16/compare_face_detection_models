from facenet_pytorch import MTCNN, InceptionResnetV1
import torch 
from torch.utils.data import DataLoader 
from torchvision import datasets
import os
import warnings 
import pickle
import cv2
import numpy as np
from PIL import Image


def mtcnn_detection(model, img, device, img_name):
    # model = MTCNN(keep_all=True, device=device)
    # print('mtcnn detecting')
    bboxes, probs, landmarks= model.detect(img, landmarks=True)
    # if not os.path.exists('./saved'):
    #     os.makedirs('./saved')
    # img_name = (img_name).split('/')[-1]
    # fw = open(os.path.join('./saved/', img_name + '.txt'), 'w')
    # for bbox, score in zip(bboxes, probs):
    #     xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    #     print(img_name)
    #     fw.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(img_name, score, xmin, ymin, xmax, ymax))
    return bboxes, probs

