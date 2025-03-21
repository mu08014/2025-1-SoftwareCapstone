import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# 1. MNIST 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 채널 차원 추가 및 정규화 (0~1)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 라벨을 one-hot 인코딩으로 변환 (클래스 개수 10)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. AlexNet 스타일 모델 구성 (MNIST에 맞게 구조 변형)
model = Sequential()

# 첫 번째 합성곱 레이어: 원래 AlexNet은 11x11 필터와 큰 스트라이드를 사용하지만 MNIST의 경우 3x3 필터 사용
model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# 두 번째 합성곱 레이어
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# 세 번째 및 네 번째 합성곱 레이어 (AlexNet의 깊이를 반영)
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# Fully Connected 레이어로 전환
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # 10개 클래스로 분류

# 3. 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 4. 모델 학습
history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# 5. 모델 평가
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])