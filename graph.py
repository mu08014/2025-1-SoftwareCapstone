import matplotlib.pyplot as plt

# 예시 데이터
epochs = list(range(1, 11))
train_loss = [1.5956, 1.5900, 1.5926 , 1.5809 , 1.5623 , 1.5278 , 1.4701 , 1.3401 , 1.3088, 1.2997]
train_accuracy = [0.2461, 0.2442, 0.3267 , 0.2800 , 0.4400 , 0.7133 , 0.6800 , 0.7733 , 0.7810, 0.7512]

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
