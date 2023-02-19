from enum import Enum
import tensorflow as tf


class TensorflowModelsForExtraction(Enum):
    ResNet50 = tf.keras.applications.ResNet50()
    VGG19 = tf.keras.applications.VGG19()
    ResNet152 = tf.keras.applications.ResNet152()


class TensorflowModelsForNormalization(Enum):
    ResNet50 = tf.keras.applications.resnet
    VGG19 = tf.keras.applications.vgg19
    ResNet152 = tf.keras.applications.resnet
