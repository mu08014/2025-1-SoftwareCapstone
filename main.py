from AlexNet_test import ExAlexNet
from Theoric_QCNN_Alexnet import ExTQCNN
from LeNet_test import ExLeNet
import os
import tensorflow as tf

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': False})
tf.config.set_visible_devices([], 'GPU')

if __name__ == '__main__':
    ExLeNet()