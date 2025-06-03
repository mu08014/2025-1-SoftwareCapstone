from qiskit import *
from AlexNet_test import PreProcessing, Train
from Theoric_QCNN_Alexnet import FastQuanv3x3_Multi, _fast_expvals, FasterQuanv3x3
from typing import Tuple, Callable

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

SHOTS = 32

class Quanv3x3LayerClass(tf.keras.layers.Layer):
    def __init__(self, kernel_size: int, channel_size: int, param_count: int=16, **kwargs,):
        super().__init__(**kwargs)
        initializer = tf.random_uniform_initializer(minval=-0.3, maxval=0.3)
        self.q_params = tf.Variable(
            initializer(shape=(kernel_size, channel_size, param_count), dtype=tf.float32),
            trainable=True
        )
        self.channel_size = channel_size
        self.kernel_size  = kernel_size
        self.output_kernel = channel_size * kernel_size
        self.param_count  = param_count
        
    def build(self, input_shape):
        if input_shape[-1] != self.channel_size:
            raise ValueError(f"Expected Cin={self.channel_size}, got {input_shape[-1]}")
        self.in_height = int(input_shape[1])
        self.in_width  = int(input_shape[2])
        super().build(input_shape)

    def call(self, x):
        y = quantum_layer(x, self.q_params, self.kernel_size)
        y.set_shape([None, self.in_height, self.in_width, self.output_kernel])
        return y

    def compute_output_shape(self, input_shape):
        b, h, w, _ = input_shape
        return (b, h, w, self.channel_size * self.kernel_size)
    
    def get_config(self):
        base = super().get_config()
        base.update(
            {
                "kernel_size":   self.kernel_size,
                "channel_size":  self.channel_size,
                "param_count":   self.param_count,
            }
        )
        return base
    
def _forward_run(x_np: np.ndarray, params_np: np.ndarray, kernel_size: int) -> np.ndarray:
    return FasterQuanv3x3(x_np, params_np, kernel_size, shots=SHOTS) #변경


'''
학습 잘 안되던 이전 코드
def _spsa_grad(x_np: np.ndarray, params_np: np.ndarray, dy_np: np.ndarray, kernel_size: int, shift: float = 0.01) -> np.ndarray:
    dx = np.random.choice([-1.0, 1.0], size=x_np.shape).astype(np.float32)
    dt = np.random.choice([-1.0, 1.0], size=params_np.shape).astype(np.float32)
    
    y_plus  = _forward_run(x_np + shift * dx, params_np + shift * dt, kernel_size)
    y_minus = _forward_run(x_np - shift * dx, params_np - shift * dt, kernel_size)
    
    Df = (y_plus - y_minus) / (2.0 * shift)
    scalar_dir = np.sum(dy_np * Df)
    
    dt_np = scalar_dir * dt
    
    return dt_np.astype(np.float32)
    
@tf.custom_gradient
def quantum_layer(x, q_params, kernel_size: int):
    y = tf.numpy_function(_forward_run, [x, q_params, kernel_size], tf.float32)

    Cin_s = x.shape[-1]
    ks_s  = q_params.shape[0]

    H_s, W_s = x.shape[1], x.shape[2]

    if None not in (Cin_s, ks_s, H_s, W_s):
        out_ch_s = Cin_s * ks_s
        tf.ensure_shape(y, [None, H_s, W_s, out_ch_s])

    def grad_fn(dy):
        if None not in (Cin_s, ks_s, H_s, W_s):
            tf.ensure_shape(dy, [None, H_s, W_s, Cin_s * ks_s])

        ks = tf.shape(q_params)[0]
        Cin = tf.shape(x)[-1]

        dy_blocks = tf.reshape(dy, (-1, H_s, W_s, Cin, ks))
        dx        = tf.reduce_sum(dy_blocks, axis=-1) / tf.cast(ks, tf.float32)

        dq = tf.numpy_function(
            _spsa_grad,
            [x, q_params, dy, kernel_size],
            tf.float32)

        dx.set_shape(x.shape)
        dq.set_shape(q_params.shape)
        return dx, dq, None

    return y, grad_fn
'''

def _spsa_grad(x_np, params_np, dy_np, kernel_size, shift=0.01):
    dx_dir = np.random.choice([-1., 1.], size=x_np.shape).astype(np.float32)
    dt_dir = np.random.choice([-1., 1.], size=params_np.shape).astype(np.float32)

    y_plus  = _forward_run(x_np + shift*dx_dir, params_np + shift*dt_dir, kernel_size)
    y_minus = _forward_run(x_np - shift*dx_dir, params_np - shift*dt_dir, kernel_size)

    df      = (y_plus - y_minus) / (2.*shift)
    scale   = np.sum(dy_np * df)

    dx_np = scale * dx_dir
    dt_np = scale * dt_dir
    return dx_np.astype(np.float32), dt_np.astype(np.float32)

@tf.custom_gradient
def quantum_layer(x, q_params, kernel_size: int):
    y = tf.numpy_function(_forward_run, [x, q_params, kernel_size], tf.float32)

    Cin_s, ks_s = x.shape[-1], q_params.shape[0]
    H_s,  W_s   = x.shape[1],  x.shape[2]
    if None not in (Cin_s, ks_s, H_s, W_s):
        tf.ensure_shape(y, [None, H_s, W_s, Cin_s*ks_s])

    def grad_fn(dy):
        if None not in (Cin_s, ks_s, H_s, W_s):
            tf.ensure_shape(dy, [None, H_s, W_s, Cin_s*ks_s])

        ks  = tf.shape(q_params)[0]
        Cin = tf.shape(x)[-1]

        dy_blocks = tf.reshape(dy, (-1, H_s, W_s, Cin, ks))
        dx        = tf.reduce_mean(dy_blocks, axis=-1)
        
        dx_spsa, dq = tf.numpy_function(
            _spsa_grad,
            [x, q_params, dy, kernel_size],
            [tf.float32, tf.float32])

        dx = dx + 0.0 * dx_spsa

        dx.set_shape(x.shape)
        dq.set_shape(q_params.shape)
        return dx, dq, None

    return y, grad_fn



def FirstQLeNet(input_shape=(14, 14, 1), num_classes=10):
    model = Sequential()
    
    model.add(Quanv3x3LayerClass(channel_size=1, kernel_size=32))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # 2nd Convolution block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # Fully connected classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def ModelCompile(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=False, jit_compile=False)
    return model

def ExFQLeNet():
    x_train, y_train, x_test, y_test = PreProcessing()
    Model = FirstQLeNet(num_classes=5)
    Model = ModelCompile(Model)
    Train(Model, x_train, y_train, x_test, y_test, save_dir='FQLeNet_Data')
    
def SecondQLeNet(input_shape=(14, 14, 1), num_classes=10):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(16, (1, 1), activation='relu', padding='same'))
    model.add(Quanv3x3LayerClass(channel_size=16, kernel_size=4))  
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # 2nd Convolution block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # Fully connected classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def ExSQLeNet():
    x_train, y_train, x_test, y_test = PreProcessing()
    Model = SecondQLeNet(num_classes=5)
    Model = ModelCompile(Model)
    Train(Model, x_train, y_train, x_test, y_test, save_dir='SQLeNet_Data')
    
def ThirdQLeNet(input_shape=(14, 14, 1), num_classes=10):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # 2nd Convolution block
    model.add(Conv2D(16, (1, 1), activation='relu', padding='same'))
    model.add(Quanv3x3LayerClass(channel_size=16, kernel_size=8))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # Fully connected classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def ExTQLeNet():
    x_train, y_train, x_test, y_test = PreProcessing()
    Model = ThirdQLeNet(num_classes=5)
    Model = ModelCompile(Model)
    Train(Model, x_train, y_train, x_test, y_test, save_dir='TQLeNet_Data')
    
def FourthQLeNet(input_shape=(14, 14, 1), num_classes=10):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # 2nd Convolution block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Quanv3x3LayerClass(channel_size=64, kernel_size=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    # Fully connected classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def ExFthQLeNet():
    x_train, y_train, x_test, y_test = PreProcessing()
    Model = FourthQLeNet(num_classes=5)
    Model = ModelCompile(Model)
    Train(Model, x_train, y_train, x_test, y_test, save_dir='FthQLeNet_Data')
    
def FSQLeNet(input_shape=(14, 14, 1), num_classes=10):
    model = Sequential()
    
    model.add(Quanv3x3LayerClass(channel_size=1, kernel_size=32))
    model.add(Quanv3x3LayerClass(channel_size=32, kernel_size=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolution block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully connected classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model
    
def ExFSQLeNet():
    x_train, y_train, x_test, y_test = PreProcessing()
    Model = FSQLeNet(num_classes=5)
    Model = ModelCompile(Model)
    Train(Model, x_train, y_train, x_test, y_test, save_dir='FSQLeNet_Data')
    
def FSTQLeNet(input_shape=(14, 14, 1), num_classes=10):
    model = Sequential()
    
    model.add(Quanv3x3LayerClass(channel_size=1, kernel_size=32))
    model.add(Quanv3x3LayerClass(channel_size=32, kernel_size=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolution block
    model.add(Quanv3x3LayerClass(channel_size=32, kernel_size=2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully connected classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model
    
def ExFSTQLeNet():
    x_train, y_train, x_test, y_test = PreProcessing()
    Model = FSTQLeNet(num_classes=5)
    Model = ModelCompile(Model)
    Train(Model, x_train, y_train, x_test, y_test, save_dir='FSTQLeNet_Data')

def FSTFQLeNet(input_shape=(14, 14, 1), num_classes=10):
    model = Sequential()
    
    model.add(Quanv3x3LayerClass(channel_size=1, kernel_size=32))
    model.add(Quanv3x3LayerClass(channel_size=32, kernel_size=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolution block
    model.add(Quanv3x3LayerClass(channel_size=32, kernel_size=2))
    model.add(Quanv3x3LayerClass(channel_size=64, kernel_size=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully connected classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model
    
def ExFSTFQLeNet():
    x_train, y_train, x_test, y_test = PreProcessing()
    Model = FSTFQLeNet(num_classes=5)
    Model = ModelCompile(Model)
    Train(Model, x_train, y_train, x_test, y_test, save_dir='FSTFQLeNet_Data')