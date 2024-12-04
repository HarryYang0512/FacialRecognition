import cv2
from keras.applications import VGG16
from keras.models import Model

base_model = VGG16(weights='imagenet', include_top=False)