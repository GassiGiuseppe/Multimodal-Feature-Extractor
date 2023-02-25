from enum import Enum
import tensorflow as tf
import torchvision.models as models

tensorflow_models_for_extraction = {
    'ResNet50': tf.keras.applications.ResNet50,
    'VGG19': tf.keras.applications.VGG19,
    'ResNet152': tf.keras.applications.ResNet152
}

tensorflow_models_for_normalization = {
    'ResNet50': tf.keras.applications.resnet,
    'VGG19': tf.keras.applications.vgg19,
    'ResNet152': tf.keras.applications.resnet
}

torch_models_for_extraction = {
    'AlexNet': models.alexnet,
    'VGG19': models.vgg19,
    'VGG11': models.vgg11
}


class TensorflowModelsForExtraction(Enum):
    ResNet50 = tf.keras.applications.ResNet50
    VGG19 = tf.keras.applications.VGG19
    ResNet152 = tf.keras.applications.ResNet152


class TensorflowModelsForNormalization(Enum):
    ResNet50 = tf.keras.applications.resnet
    VGG19 = tf.keras.applications.vgg19
    ResNet152 = tf.keras.applications.resnet


class TorchModelsForExtraction(Enum):
    AlexNet = models.alexnet
    VGG11 = models.vgg11
    VGG19 = models.vgg19
