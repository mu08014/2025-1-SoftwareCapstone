import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model

from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from sklearn.metrics import accuracy_score

import os
import matplotlib.pyplot as plt
import numpy as np

MNIST_data_size = 500
TEST_data_size = 150
EPOCH = 10


def PreProcessing():
    (x_train_all, y_train_all), (x_test_all, y_test_all) = mnist.load_data()

    selected_classes = [0, 2, 3, 5, 8]  # 사용할 클래스 인덱스
    class_mapping = {label: idx for idx, label in enumerate(selected_classes)}
    train_per_class = MNIST_data_size // len(selected_classes)
    test_per_class = TEST_data_size // len(selected_classes)

    x_train, y_train, x_test, y_test = [], [], [], []

    for cls in selected_classes:
        cls_train_idx = np.where(y_train_all == cls)[0]
        np.random.shuffle(cls_train_idx)
        cls_train_idx = cls_train_idx[:train_per_class]

        x_train.append(x_train_all[cls_train_idx])
        y_train.append(np.full(train_per_class, class_mapping[cls]))

        cls_test_idx = np.where(y_test_all == cls)[0]
        np.random.shuffle(cls_test_idx)
        cls_test_idx = cls_test_idx[:test_per_class]

        x_test.append(x_test_all[cls_test_idx])
        y_test.append(np.full(test_per_class, class_mapping[cls]))

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    train_indices = np.random.permutation(len(x_train))
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    test_indices = np.random.permutation(len(x_test))
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    x_train = tf.image.resize(x_train, [14, 14], method='bilinear')
    x_test = tf.image.resize(x_test, [14, 14], method='bilinear')

    y_train = to_categorical(y_train, 5)
    y_test = to_categorical(y_test, 5)

    return x_train, y_train, x_test, y_test


def AlexNet():
    model = Sequential()

    # Conv Layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                     input_shape=(14, 14, 1)))
    model.add(BatchNormalization())
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

def ModelCompile(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

class ClasswiseAccuracyLogger(tf.keras.callbacks.Callback):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_class_accuracies = [[] for _ in range(y_train.shape[1])]
        self.test_class_accuracies = [[] for _ in range(y_test.shape[1])]

    def on_epoch_end(self, epoch, logs=None):
        pred_train = np.argmax(self.model.predict(self.x_train, verbose=0), axis=1)
        true_train = np.argmax(self.y_train, axis=1)
        pred_test = np.argmax(self.model.predict(self.x_test, verbose=0), axis=1)
        true_test = np.argmax(self.y_test, axis=1)

        for cls in range(self.y_train.shape[1]):
            train_indices = np.where(true_train == cls)[0]
            test_indices = np.where(true_test == cls)[0]

            train_acc = accuracy_score(true_train[train_indices], pred_train[train_indices]) if len(train_indices) > 0 else 0
            test_acc = accuracy_score(true_test[test_indices], pred_test[test_indices]) if len(test_indices) > 0 else 0

            self.train_class_accuracies[cls].append(train_acc)
            self.test_class_accuracies[cls].append(test_acc)

def Train(model, x_train, y_train, x_test, y_test, save_dir='AlexNet_Data'):
    logger = LrLogger()
    classwise_logger = ClasswiseAccuracyLogger(x_train, y_train, x_test, y_test)

    model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=EPOCH,
        validation_data=(x_test, y_test),
        callbacks=[logger, classwise_logger]
    )

    epochs = range(1, EPOCH + 1)
    
    graph_path = os.path.join(save_dir, 'fq_lenet_model')
    model.save(graph_path, save_format="tf")

    # cal FLOPs
    @tf.function
    def model_fn(x):
        return model(x)

    input_tensor = tf.random.normal([1, 14, 14, 1])
    concrete_func = model_fn.get_concrete_function(input_tensor)

    profiler = model_analyzer.Profiler(graph=concrete_func.graph)
    opts = ProfileOptionBuilder.float_operation()

    FLOPs = profiler.profile_operations(options=opts)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_graph_path = os.path.join(save_dir, 'model.png')
    plot_model(model, to_file=model_graph_path,show_shapes=True,show_layer_names=True,dpi=96)

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

    graph_path = os.path.join(save_dir, 'train_graph.png')
    plt.savefig(graph_path)

    num_classes = y_test.shape[1]

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

    log_path = os.path.join(save_dir, 'training_log.txt')
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


    pred_test = np.argmax(model.predict(x_test, verbose=0), axis=1)
    true_test = np.argmax(y_test, axis=1)

    print("예측된 클래스 분포:", np.bincount(pred_test))
    print("실제 정답 클래스 분포:", np.bincount(true_test))

class LrLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.accuracy = []
        self.loss = []
        self.val_accuracy = []
        self.val_loss = []
        self.params = 0

    def on_epoch_end(self, epoch, logs=None):
        lr = logs.get('accuracy')
        ls = logs.get('loss')
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.val_loss.append(logs.get('val_loss'))

        self.accuracy.append(lr)
        self.loss.append(ls)
        self.params = max(self.params, self.model.count_params())

def ExAlexNet():
    x_train, y_train, x_test, y_test = PreProcessing()
    Model = AlexNet()
    Model = ModelCompile(Model)
    Train(Model, x_train, y_train, x_test, y_test)