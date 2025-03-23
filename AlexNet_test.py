import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

import os
import matplotlib.pyplot as plt

MNIST_data_size = 1000
EPOCH = 25

def PreProcessing():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train[:MNIST_data_size]
    y_train = y_train[:MNIST_data_size]

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

def AlexNet():
    model = Sequential()

    # Conv Layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                     input_shape=(28, 28, 1)))
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
    model.add(Dense(10, activation='softmax'))

    return model

def ModelCompile(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def Train(model, x_train, y_train, x_test, y_test):
    logger = LrLogger()
    model.fit(x_train, y_train, batch_size=128, epochs=EPOCH, validation_data=(x_test, y_test), callbacks=[logger])
    epochs = range(1, len(logger.accuracy) + 1)

    #draw plot
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
    
    #save file
    save_dir = 'AlexNet_Data'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    graph_path = os.path.join(save_dir, 'training_graph.png')
    plt.savefig(graph_path)
    plt.show()
    
    log_path = os.path.join(save_dir, 'training_log.txt')
    with open(log_path, 'w') as f:
        f.write(f"{MNIST_data_size} MNIST Data size, {EPOCH} epochs\n\n")
        f.write("Epoch\tAccuracy\tLoss\n")
        for i in range(len(epochs)):
            f.write(f"{epochs[i]}\t{logger.accuracy[i]}\t{logger.loss[i]}\n")
        f.write(f"\nMax Parameter Count: {logger.params}\n")
    
    print('Max parameter:', logger.params)
    print(f'Graph saved to: {graph_path}')
    print(f'Log saved to: {log_path}')


class LrLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.accuracy = []
        self.loss = []
        self.params = 0

    def on_epoch_end(self, epoch, logs=None):
        lr = logs.get('accuracy')
        ls = logs.get('loss')

        self.accuracy.append(lr)
        self.loss.append(ls)
        self.params = max(self.params, self.model.count_params())

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = PreProcessing()
    Model = AlexNet()
    Model = ModelCompile(Model)
    Train(Model, x_train, y_train, x_test, y_test)