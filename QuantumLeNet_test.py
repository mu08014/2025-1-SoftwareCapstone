from qiskit import *
from AlexNet_test import PreProcessing, Train
from Theoric_QCNN_Alexnet import FastQuanv3x3_Multi, _fast_expvals
from typing import Tuple, Callable

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

SHOTS = 32

class Quanv3x3LayerClass(tf.keras.layers.Layer):
    def __init__(self, kernel_size: int, channel_size: int, param_count: int=16):
        super().__init__()
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
    
def _forward_run(x_np: np.ndarray, params_np: np.ndarray, kernel_size: int) -> np.ndarray:
    return FastQuanv3x3_Multi(x_np, params_np, kernel_size, shots=SHOTS)

def _spsa_grad(x_np: np.ndarray, params_np: np.ndarray, dy_np: np.ndarray, kernel_size: int, shift: float = 0.01) -> np.ndarray:
    dx = np.random.choice([-1.0, 1.0], size=x_np.shape).astype(np.float32)
    dt = np.random.choice([-1.0, 1.0], size=params_np.shape).astype(np.float32)
    
    y_plus  = _forward_run(x_np + shift * dx, params_np + shift * dt, kernel_size)
    y_minus = _forward_run(x_np - shift * dx, params_np - shift * dt, kernel_size)
    
    Df = (y_plus - y_minus) / (2.0 * shift)
    scalar_dir = np.sum(dy_np * Df)
    
    dx_np = scalar_dir * dx
    dt_np = scalar_dir * dt
    
    return dx_np.astype(np.float32), dt_np.astype(np.float32)


@tf.custom_gradient
def quantum_layer(x, q_params, kernel_size: int):
    y = tf.numpy_function(_forward_run, [x, q_params, kernel_size], tf.float32)
    
    ks = int(q_params.shape[0])
    cs = int(q_params.shape[1])
    out_ch = ks * cs
    tf.ensure_shape(y, [None, 14, 14, out_ch])

    def grad_fn(dy):
        dx_dq = tf.numpy_function(_spsa_grad,
                                  [x, q_params, dy, kernel_size],
                                  [tf.float32, tf.float32])
        
        dx, dq = dx_dq
        dx.set_shape(x.shape)
        dq.set_shape(q_params.shape)
        return dx, dq, None

    return y, grad_fn

'''

@tf.custom_gradient
def quantum_layer(x: tf.Tensor,           # (B,H,W,C)
                  q_params: tf.Tensor,    # (K,C,16)
                  kernel_size: tf.Tensor  # scalar int32
                 ) -> Tuple[tf.Tensor, Callable]:

    ks = 3
    padding = 'VALID'          # 12×12 중심부만 추출
    patches = tf.image.extract_patches(
        images=x,
        sizes  =[1, ks, ks, 1],
        strides=[1, 1, 1, 1],
        rates  =[1, 1, 1, 1],
        padding=padding
    )                           # (B, Fx, Fy, ks*ks*C)

    B   = tf.shape(x)[0]
    Fx  = tf.shape(patches)[1]                # H-ks+1 = 12
    Fy  = tf.shape(patches)[2]                # W-ks+1 = 12
    Cin = tf.shape(x)[-1]
    K   = tf.shape(q_params)[0]

    # ---------- 1) 채널별로 분리해 (B*Fx*Fy*Cin, 9) ----------
    patches = tf.reshape(patches, (B, Fx, Fy, Cin, ks*ks))
    patches = tf.reshape(patches, (B*Fx*Fy*Cin, ks*ks))

    # ---------- 2) 각 patch 에 대해 K개의 kernel 매핑 ----------
    patches = tf.repeat(patches, repeats=K, axis=0)             # (B*Fx*Fy*Cin*K, 9)

    theta_per_kc = tf.reshape(q_params, (K*Cin, 16))            # (K*Cin, 16)
    thetas = tf.tile(theta_per_kc, [B*Fx*Fy, 1])                 # (B*Fx*Fy*K*Cin, 16)

    # ---------- 3) GPU 한 번 호출로 expval ----------
    expvals = tf.numpy_function(_fast_expvals, [patches, thetas], tf.float32)
    expvals = tf.reshape(expvals, (B, Fx, Fy, Cin, K))          # (B,12,12,Cin,K)
    expvals = tf.transpose(expvals, [0, 1, 2, 4, 3])            # (B,12,12,K,Cin)
    expvals = tf.reshape(expvals, (B, Fx, Fy, K*Cin))           # (B,12,12,K*Cin)

    # ---------- 4) 가장자리 1칸씩 0-패딩 → 14×14 --------------
    y = tf.pad(expvals,
               paddings=[[0, 0],       # batch
                         [1, 1],       # height: top, bottom
                         [1, 1],       # width : left, right
                         [0, 0]],      # channels
               constant_values=0.0)

    # 최종 shape 명시
    out_shape = (None,                       # batch 미정
                 x.shape[1],                 # H (정적: 14)
                 x.shape[2],                 # W (정적: 14)
                 q_params.shape[0] * x.shape[-1])  # K*Cin
    y.set_shape(out_shape)

    # ---------- 5) SPSA gradient ------------------------
    def grad_fn(dy):
        dq = tf.numpy_function(
            _spsa_grad,
            [x, q_params, dy, kernel_size],
            tf.float32
        )
        dq.set_shape(q_params.shape)
        return tf.zeros_like(x), dq, None    # dx≈0, dkern 반환

    return y, grad_fn
'''

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