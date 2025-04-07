from qiskit import *
from AlexNet_test import EPOCH
from AlexNet_test import PreProcessing
from AlexNet_test import LrLogger
from AlexNet_test import MNIST_data_size
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

SHOTS = 128

def Encoding2x2(qc: QuantumCircuit, data: list):
    for i in range(4):
        qc.rx(2*np.pi*data[i]-np.pi, i)
    return qc

def Ansatz2x2(qc: QuantumCircuit, train_theta: list):
    #depth : 2 x log(4)
    qc.crz(train_theta[0], 1, 0)
    qc.crx(train_theta[1], 1, 0)
    qc.crz(train_theta[2], 3, 2)
    qc.crx(train_theta[3], 3, 2)
    qc.crz(train_theta[4], 2, 0)
    qc.crx(train_theta[5], 2, 0)

    return qc

def Quanv2x2LayerCircuit(input_data: list, train_theta: list, shots=SHOTS):
    qc = QuantumCircuit(4, 1)
    qc = Encoding2x2(qc, input_data)
    qc = Ansatz2x2(qc, train_theta)
    qc.measure(0, 0)

    avgZ = QuanvAerMeasure(qc, shots)

    return avgZ

def Encoding3x3(qc: QuantumCircuit, data: np.array):
    for i in range(9):
        qc.rx(2*np.pi*data[i]-np.pi, i)
    return qc

def Ansatz3x3(qc: QuantumCircuit, train_theta: np.array):
    # depth : 2 x log(9)
    qc.crz(train_theta[0], 1, 0)
    qc.crx(train_theta[1], 1, 0)
    qc.crz(train_theta[2], 3, 2)
    qc.crx(train_theta[3], 3, 2)
    qc.crz(train_theta[4], 2, 0)
    qc.crx(train_theta[5], 2, 0)

    qc.crz(train_theta[6], 8, 7)
    qc.crx(train_theta[7], 8, 7)

    qc.crz(train_theta[8], 5, 4)
    qc.crx(train_theta[9], 5, 4)
    qc.crz(train_theta[10], 7, 6)
    qc.crx(train_theta[11], 7, 6)
    qc.crz(train_theta[12], 6, 4)
    qc.crx(train_theta[13], 6, 4)

    qc.crz(train_theta[14], 4, 0)
    qc.crx(train_theta[15], 4, 0)

    return qc

def Quanv3x3LayerCircuit(input_data: np.array, train_theta: np.array, shots=SHOTS):
    qc = QuantumCircuit(9, 1)
    qc = Encoding3x3(qc, input_data)
    qc = Ansatz3x3(qc, train_theta)
    qc.measure(0, 0)

    avgZ = QuanvAerMeasure(qc, shots)

    #회로 그림 알고싶을 때 주석 제거
    #qc.draw('mpl')
    #plt.show()

    return avgZ

def QuanvAerMeasure(qc: QuantumCircuit, shots=SHOTS):
    backend = Aer.get_backend(name='aer_simulator')
    counts = backend.run(qc, shots=shots).result().get_counts()

    #Measure data 알고 싶을 때 주석 제거
    #plot_histogram(counts)
    #plt.show()

    val0 = counts.get('0', 0)
    val1 = counts.get('1', 0)

    return (val0 - val1) / SHOTS # <Z> expectation value

def Quanv2x2Layer(data: np.array, params: np.array, stride: int, shots=SHOTS):
    #파라미터 수 확인
    if len(params) != 6:
        return False
    #stride가 적절한지 확인
    if (len(data) - 2) % stride != 0 or (len(data[0]) - 2) % stride != 0:
        return False

    feat_map = np.zeros((int((len(data) - 2) / stride) + 1, int((len(data[0]) - 2) / stride) + 1), dtype=np.float32)
    idx_r = 0
    for r in range(0, len(data), stride):
        idx_c = 0
        for c in range(0, len(data[0]), stride):
            patch = data[r:r + 2, c:c + 2].flatten()
            avgZ = Quanv2x2LayerCircuit(patch, params)
            feat_map[idx_r, idx_c] = avgZ
            idx_c += 1
        idx_r += 1
    return feat_map

def Quanv3x3Layer(data: np.array, params: np.array, stride: int, shots=SHOTS):
    # 파라미터 수 확인
    if len(params) != 16:
        return False
    # stride가 적절한지 확인
    if (len(data) - 3) % stride != 0 or (len(data[0]) - 3) % stride != 0:
        return False

    fx = int((len(data) - 3) / stride) + 1
    fy = int((len(data[0]) - 3) / stride) + 1
    feat_map = np.zeros((fx, fy), dtype=np.float32)
    idx_r = 0
    for r in range(0, fx, stride):
        idx_c = 0
        for c in range(0, fy, stride):
            patch = data[r:r + 3, c:c + 3].flatten()
            avgZ = Quanv3x3LayerCircuit(patch, params)
            feat_map[idx_r, idx_c] = avgZ
            idx_c += 1
        idx_r += 1
    return feat_map

def BatchQuanv2x2(data: np.array, params: np.array, channel_size: int, shots=SHOTS):
    B = len(data)                       #Bx28x28xk
    avg_vals = np.mean(data, axis=-1)   #Bx28x28

    out = []
    for i in range(B):
        feat_stack = []
        for j in range(channel_size):
            feat_map = Quanv2x2Layer(avg_vals[i], params[j], stride=1, shots=shots)         #27x27
            feat_map_uint8 = ((feat_map + 1) / 2).astype(np.float32)
            feat_map_padded = np.pad(feat_map_uint8, pad_width=((0, 1), (0, 1)), mode='constant') #28x28
            feat_stack.append(feat_map_padded)                                              #channel_sizex28x28
        out.append(np.transpose(feat_stack, (1, 2, 0)))                                 #28x28xchannel_size
    return np.stack(out, axis=0)                                                            #Bxchannel_sizex28x28

def BatchQuanv3x3(data: np.array, params: np.array, channel_size: int, shots=SHOTS):
    B = len(data)                       #Bx28x28xk
    avg_vals = np.mean(data, axis=-1)   #Bx28x28

    out = []
    for i in range(B):
        feat_stack = []
        print(f'{i}th Image is running...')
        for j in range(channel_size):
            feat_map = Quanv3x3Layer(avg_vals[i], params[j], stride=1, shots=shots)         #26x26
            feat_map_uint8 = ((feat_map + 1) / 2).astype(np.float32)
            feat_map_padded = np.pad(feat_map_uint8, pad_width=1, mode='constant')          #28x28
            feat_stack.append(feat_map_padded)                                              #channel_sizex28x28
        out.append(np.transpose(feat_stack, (1, 2, 0)))                                 #28x28xchannel_size
    return np.stack(out, axis=0)                                                            #Bxchannel_sizex28x28

def MakeQParams(channel_size: int, param_count: int):
    return np.random.uniform(-math.pi, math.pi, size=(channel_size, param_count))

class Quanv3x3LayerClass(tf.keras.layers.Layer):
    def __init__(self, channel_size, param_count):
        super().__init__()
        initializer = tf.random_uniform_initializer(minval=-np.pi, maxval=np.pi)
        self.q_params = tf.Variable(
            initial_value=initializer(shape=(channel_size, param_count), dtype="float32"),
            trainable=True
        )
        self.channel_size = channel_size
        self.param_count = param_count

    def call(self, x):
        ql = quantum_layer(x, self.q_params)
        output_shape = (x.shape[0], x.shape[1], x.shape[2], self.channel_size)
        return tf.ensure_shape(ql, output_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.channel_size,)

@tf.custom_gradient
def quantum_layer(x, q_params):
    def run_quantum_circuit(x_np, params_np):
        return BatchQuanv3x3(x_np, params_np, channel_size=params_np.shape[0], shots=SHOTS)

    y = tf.numpy_function(run_quantum_circuit, [x, q_params], tf.float32)

    def grad(dy):
        grad_params = np.ones_like(q_params.shape) * 0.01
        grad_q_params = tf.reduce_mean(dy) * tf.ones_like(q_params, dtype=tf.float32)
        return None, grad_q_params

    return y, grad


def OneQLayerFourCLayer():
    model = Sequential()
    model.add(Quanv3x3LayerClass(64, 16))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

def ModelCompile(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def Train(model, x_train, y_train, x_test, y_test):
    logger = LrLogger()
    model.fit(x_train, y_train, epochs=EPOCH, validation_data=(x_test, y_test), verbose=1, callbacks=[logger])
    epochs = range(1, len(logger.accuracy) + 1)

    # draw plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, logger.accuracy, marker='o')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, logger.loss, marker='o')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()

    # cal FLOPs
    @tf.function
    def model_fn(x):
        return model(x)

    input_tensor = tf.random.normal([1, 14, 14, 1])
    concrete_func = model_fn.get_concrete_function(input_tensor)

    profiler = model_analyzer.Profiler(graph=concrete_func.graph)
    opts = ProfileOptionBuilder.float_operation()

    FLOPs = profiler.profile_operations(options=opts)

    # save file
    save_dir = 'OneQLayer_Data'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    graph_path = os.path.join(save_dir, 'training_graph.png')
    plt.savefig(graph_path)

    log_path = os.path.join(save_dir, 'training_log.txt')
    model_path = os.path.join(save_dir, 'model.png')

    with open(log_path, 'w') as f:
        f.write(f"{MNIST_data_size} MNIST Data size, {EPOCH} epochs\n\n")
        f.write("Epoch\tAccuracy\tLoss\n")
        for i in range(len(epochs)):
            f.write(f"{epochs[i]}\t{logger.accuracy[i]}\t{logger.loss[i]}\n")
        f.write(f"\nMax Parameter Count: {logger.params}\n")
        f.write(f"\nTotal FLOPs : {FLOPs}\n")

    plot_model(model, to_file=model_path, show_shapes=True, show_layer_names=True)

    print(f'FLOPs: {FLOPs}')
    print('Max parameter:', logger.params)
    print(f'Graph saved to: {graph_path}')
    print(f'Log saved to: {log_path}')


def ExTQCNN():
    x_train, y_train, x_test, y_test = PreProcessing()
    Model = OneQLayerFourCLayer()
    Model = ModelCompile(Model)
    Train(Model, x_train, y_train, x_test, y_test)














