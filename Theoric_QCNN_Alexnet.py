from qiskit import *
from qiskit.circuit import ParameterVector

from AlexNet_test import EPOCH
from AlexNet_test import PreProcessing
from AlexNet_test import LrLogger
from AlexNet_test import ClasswiseAccuracyLogger
from AlexNet_test import MNIST_data_size
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import math
import os
import time
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from concurrent.futures import ThreadPoolExecutor

SHOTS = 32
USE_GPU = False

GPU_BACKEND_OPTS = dict(
    method="statevector",
    device="GPU",
    precision="single",
    runtime_parameter_bind_enable=True,
    batched_shots_gpu=True,
    max_parallel_threads=1,
    max_parallel_experiments=1
)

CPU_BACKEND_OPTS = dict(
    method="statevector",
    device="CPU",
    precision="single",
    max_parallel_threads=8,
    max_parallel_experiments=8,
    max_parallel_shots=1
)

backend_opts = GPU_BACKEND_OPTS if USE_GPU else CPU_BACKEND_OPTS

def Encoding3x3(qc: QuantumCircuit, data: np.array):
    for i in range(9):
        angle = np.pi * data[i]
        qc.rx(angle, i)
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

def Quanv3x3LayerCircuit(input_data: np.array, train_theta: np.array):
    qc = QuantumCircuit(9, 1)
    qc = Encoding3x3(qc, input_data)
    qc = Ansatz3x3(qc, train_theta)
    qc.measure(0, 0)

    #회로 그림 알고싶을 때 주석 제거
    #qc.draw('mpl')
    #plt.show()

    return qc


def QuanvAerMeasure(qc: list[QuantumCircuit], shots=SHOTS):
    # backend = AerSimulator()
    # transpiled = transpile(qc, backend)
    # results = backend.run(transpiled, shots=SHOTS).result()

    # 멀티스레딩
    backend = AerSimulator(method='statevector',
                           max_parallel_threads=8,
                           max_parallel_experiments=8,
                           max_parallel_shots=1)

    transpiled = transpile(qc, backend)
    results = backend.run(transpiled, shots=shots).result()

    return results

def QuanvGPUAerMeasure(qc: list[QuantumCircuit], shots=SHOTS):
    backend = AerSimulator(method="statevector", device="GPU", precision="single",
                        max_parallel_threads=8, max_parallel_experiments=8, max_parallel_shots=1)
    transpiled = transpile(qc, backend)
    results = backend.run(transpiled, shots=shots).result()

    return results

def QuanvGPUBatchMeasure(patches, thetas, shots=SHOTS):
    backend, transpiled, d, t = QCTemplate()

    bound_circuits = []
    for patch, theta in zip(patches, thetas):
        bind_map = {d[i]: float(patch[i]) for i in range(9)}
        bind_map.update({t[j]: float(theta[j]) for j in range(16)})
        bound_circuits.append(transpiled.assign_parameters(bind_map, inplace=False))

    job = backend.run(bound_circuits, shots=shots)
    return job.result()

@lru_cache(maxsize=1)
def QCTemplate():
    d = ParameterVector('d', 9)
    t = ParameterVector('t', 16)
    qc_template = QuantumCircuit(9, 1)
    
    for i in range(9):
        qc_template.rx(np.pi * d[i], i)
    
    qc_template.crz(t[0],  1, 0); qc_template.crx(t[1],  1, 0)
    qc_template.crz(t[2],  3, 2); qc_template.crx(t[3],  3, 2)
    qc_template.crz(t[4],  2, 0); qc_template.crx(t[5],  2, 0)
    qc_template.crz(t[6],  8, 7); qc_template.crx(t[7],  8, 7)
    qc_template.crz(t[8],  5, 4); qc_template.crx(t[9],  5, 4)
    qc_template.crz(t[10], 7, 6); qc_template.crx(t[11], 7, 6)
    qc_template.crz(t[12], 6, 4); qc_template.crx(t[13], 6, 4)
    qc_template.crz(t[14], 4, 0); qc_template.crx(t[15], 4, 0)

    qc_template.save_probabilities([0], label='p0')

    backend = AerSimulator(**backend_opts)

    transpiled_template = transpile(qc_template, backend, optimization_level=3)
    
    return backend, transpiled_template, d, t


MAX_CIRCS_PER_RUN = 4096


def _make_binds(mat, d, t):
    keys = list(d) + list(t)
    return [ {k: [float(v)] for k, v in zip(keys, row)} for row in mat ]


def QuanvBatchProbabilities(patches, thetas):
    '''
    backend, transpiled, d, t = QCTemplate()
    total, exps = len(patches), []

    for s in range(0, total, MAX_CIRCS_PER_RUN):
        e      = min(s + MAX_CIRCS_PER_RUN, total)
        binds  = _make_binds(np.hstack([patches[s:e], thetas[s:e]]), d, t)
        circs  = [transpiled] * len(binds)

        res = backend.run(circs, parameter_binds=binds, shots=1).result()
        exps.extend((r.data.p0[0] - r.data.p0[1] + 1) * 0.5 for r in res.results)
    '''
    
    backend, tqc, d, t = QCTemplate()
    param_mat = np.hstack([patches, thetas]).astype(np.float32)
    res = backend.run(
        tqc,
        parameter_binds=param_mat,
        shots=1,
    ).result()
    
    return ( (r.data.p0[0] - r.data.p0[1] + 1) * 0.5 for r in res.results )

def _extract_patches_tf(x):
    ks = 3
    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, ks, ks, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    B = tf.shape(x)[0]
    patches = tf.reshape(patches, (B, -1, ks*ks, tf.shape(x)[-1]))
    return patches


def _fast_expvals(patches_np: np.ndarray, thetas_np:  np.ndarray) -> np.ndarray:
    print(f"start CPU batch : {patches_np.shape}")
    tic = time.perf_counter()
    
    expvals = np.asarray(
        QuanvBatchProbabilities(
            patches_np.astype(np.float32),
            thetas_np.astype(np.float32)
        ),
        dtype=np.float32
    )
    
    toc = time.perf_counter()
    n = len(patches_np)
    print(f"[Quanv] GPU batch {n:7d} patch → {((toc-tic)*1000):6.1f} ms")
    
    return expvals


def QuanvBatchExpectation(patches: np.ndarray, thetas:  np.ndarray) -> list[float]:
    backend, transpiled, d, t = QCTemplate()
    bound = []
    for patch, theta in zip(patches, thetas):
        bind_map = {d[i]: float(patch[i])   for i in range(9)}
        bind_map.update({t[j]: float(theta[j]) for j in range(16)})
        bound.append(transpiled.assign_parameters(bind_map, inplace=False))

    job = backend.run(bound).result()
    exps = []
    dim = 2**9
    for idx in range(len(bound)):
        sv = job.get_statevector(idx)
        p0 = 0.0
        p1 = 0.0

        for amp_index, amp in enumerate(sv):
            prob = abs(amp)**2
            if (amp_index & 1) == 0:
                p0 += prob
            else:
                p1 += prob
        exps.append((p0 - p1 + 1.0) * 0.5)
    return exps


@lru_cache(maxsize=8)
def template_clones(n):
    tqc = QCTemplate()[1]
    return [tqc] * n


def FastQuanv3x3_Multi(data: np.ndarray, params: np.ndarray, kernel_size: int, stride: int = 1, shots: int=SHOTS, chunk_size: int=4096) -> np.ndarray:
    B, H, W, C_in = data.shape
    fx = (H - 3) // stride + 1
    fy = (W - 3) // stride + 1
    
    tf.print("[Quanv] batches:", B, "  patches per batch:", fx * fy * C_in)
    
    out = np.zeros((B, H, W, C_in * kernel_size), dtype=np.float32)
    
    for b in range(B):
        patches = []
        thetas = []
        meta = []
        for k in range(kernel_size):
            for r in range(fx):
                for y in range(fy):
                    for c in range(C_in):
                        patch = data[b,
                                r * stride:r * stride + 3,
                                y * stride:y * stride + 3,
                                c].flatten()
                        patches.append(patch)
                        thetas.append(params[k, c])
                        meta.append((k, r, y, c))
        exp_vals = []
        for i in range(0, len(patches), chunk_size):
            exp_vals.extend(
                QuanvBatchProbabilities(
                    np.array(patches[i:i+chunk_size], dtype=np.float32),
                    np.array(thetas[i:i+chunk_size], dtype=np.float32)
                )
            )

        for idx, (k, r, y, c) in enumerate(meta):
            out[b, r + 1, y + 1, k * C_in + c] = exp_vals[idx]
    
    return out


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

#back propagation에서 앞 classical Layer까지 흘려주기 위해 구조 변경
@tf.custom_gradient
def quantum_layer(x, q_params):

    def forward_run(x_np, params_np):
        return FastQuanv3x3_Multi(x_np, params_np, 32, shots=SHOTS)
    
    y = tf.numpy_function(forward_run, [x, q_params], tf.float32)

    def grad_fn(dy):
        shift  = 0.01
        Delta  = np.random.choice([-1.0, 1.0], size=q_params.shape)
        y_plus  = tf.numpy_function(forward_run, [x, q_params + shift*Delta], tf.float32)
        y_minus = tf.numpy_function(forward_run, [x, q_params - shift*Delta], tf.float32)
        partial     = (y_plus - y_minus) / (2.0 * shift)
        chain_mult  = tf.reduce_sum(dy * partial)
        dq          = chain_mult * Delta

        dx_mean = tf.reduce_mean(dy, axis=-1, keepdims=True)

        cin = tf.shape(x)[-1]
        dx   = tf.tile(dx_mean, [1, 1, 1, cin])

        return dx, tf.reshape(dq, tf.shape(q_params))

    y.set_shape((None, None, None, q_params.shape[0]))
    
    return y, grad_fn

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
    model.add(Dense(5, activation='softmax'))

    return model

def OneCOneQThreeCLayer():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                     input_shape=(14, 14, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(16, (1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Quanv3x3LayerClass(128, 16))
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
    model.add(Dense(5, activation='softmax'))

    return model

def ModelCompile(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
    return model

def Train(model, x_train, y_train, x_test, y_test):
    logger = LrLogger()
    model.fit(x_train, y_train, epochs=EPOCH, validation_data=(x_test, y_test), verbose=1, callbacks=[logger])
    epochs = range(1, len(logger.accuracy) + 1)
    classwise_logger = ClasswiseAccuracyLogger(x_train, y_train, x_test, y_test)

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

    # draw plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, logger.accuracy, marker='o')
    plt.title("Train Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, logger.loss, marker='o')
    plt.title("Train Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()

    graph_path = os.path.join(save_dir, 'training_graph.png')
    plt.savefig(graph_path)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, logger.val_accuracy, marker='o', color='tab:orange')
    plt.title("Test Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, logger.val_loss, marker='o', color='tab:red')
    plt.title("Test Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.tight_layout()

    test_graph_path = os.path.join(save_dir, 'test_graph.png')
    plt.savefig(test_graph_path)

    log_path = os.path.join(save_dir, 'training_log.txt')
    model_path = os.path.join(save_dir, 'model.png')

    num_classes = int(np.max(y_test)) + 1

    plt.figure()
    for cls in range(num_classes):
        plt.plot(epochs, classwise_logger.test_class_accuracies[cls], marker='o', label=f'Class {cls}')
    plt.title("Test Class-wise Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    test_class_graph_path = os.path.join(save_dir, 'test_classwise_accuracy_graph.png')
    plt.savefig(test_class_graph_path)

    plt.figure()
    for cls in range(num_classes):
        plt.plot(epochs, classwise_logger.train_class_accuracies[cls], marker='o', label=f'Class {cls}')
    plt.title("Train Class-wise Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    train_class_graph_path = os.path.join(save_dir, 'train_classwise_accuracy_graph.png')
    plt.savefig(train_class_graph_path)

    with open(log_path, 'w') as f:
        f.write(f"{MNIST_data_size} MNIST Data size, {EPOCH} epochs\n\n")
        f.write("Epoch\tAccuracy\tLoss\n")
        for i in range(len(epochs)):
            f.write(f"{epochs[i]}\t{logger.accuracy[i]}\t{logger.loss[i]}\n")

        f.write("\nTrain Class-wise Accuracy per Epoch:\n")
        for cls in range(num_classes):
            f.write(f"Class {cls}: {classwise_logger.train_class_accuracies[cls]}\n")

        f.write("\nTest Class-wise Accuracy per Epoch:\n")
        for cls in range(num_classes):
            f.write(f"Class {cls}: {classwise_logger.test_class_accuracies[cls]}\n")

        f.write(f"\nMax Parameter Count: {logger.params}\n")
        f.write(f"\nTotal FLOPs : {FLOPs}\n")

    plot_model(model, to_file=model_path, show_shapes=True, show_layer_names=True)

    print(f'FLOPs: {FLOPs}')
    print('Max parameter:', logger.params)
    print(f'Graph saved to: {graph_path}')
    print(f'Log saved to: {log_path}')


def ExTQCNN():
    x_train, y_train, x_test, y_test = PreProcessing()
    Model = OneCOneQThreeCLayer()
    Model = ModelCompile(Model)
    Train(Model, x_train, y_train, x_test, y_test)














