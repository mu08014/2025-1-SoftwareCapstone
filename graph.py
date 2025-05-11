import matplotlib.pyplot as plt

# 예시 데이터
epochs = list(range(1, 10))
train_loss = [1.6056, 1.6100, 1.6126 , 1.6109 , 1.6097 , 1.6100 , 1.6119 , 1.6095 , 1.6088]
train_accuracy = [0.2000, 0.1960, 0.1975 , 0.1900 , 0.1973 , 0.2000 , 0.2000 , 0.2000 , 0.2067]

# 서브플롯 생성
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Train Loss 그래프
axs[0].plot(epochs, train_loss, 'b-o')
axs[0].set_title('Test Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)

# Train Accuracy 그래프
axs[1].plot(epochs, train_accuracy, 'r-s')
axs[1].set_title('Test Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)

# 레이아웃 정리 및 표시
plt.tight_layout()
plt.show()
