from AlexNet_test import ExAlexNet
from Theoric_QCNN_Alexnet import ExTQCNN
from LeNet_test import ExLeNet
from QuantumLeNet_test import ExFQLeNet, ExSQLeNet, ExTQLeNet
import os

os.environ["OMP_NUM_THREADS"] = "12"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': False})

JIT_KW = dict(jit_compile=False)

if __name__ == '__main__':
    #ExTQLeNet()
    ExSQLeNet()
    #ExLeNet()