# 2025-1-SoftwareCapstone

본 연구는 LeNet 계열의 합성곱 신경망(CNN)을 기반으로, 양자–고전 하이브리드 신경망 구조에서 양자 레이어의 삽입 위치 및 삽입 개수가 전체 학습 성능에 미치는 영향을 실험적으로 규명한다. 입력단, 중간단, 출력단 전 구간에 대해 교차 실험을 수행한 결과, “가정 1. Quantum Layer의 교체 위치에 따라 전체 Quantum Classical Hybrid Neural Network 구조의 학습 성능이 달라진다.”와 “가정 2. Quantum Layer의 교체 개수에 따라 전체 Quantum Classical Hybrid Neural Network 구조의 학습 성능이 달라진다.”를 검증하고자 하는 실험에 대한 파이썬 코드이며, TensorFlow 프레임워크의 Custom Layer 기능을 활용하여, 고전적 합성곱 신경망(CNN) 구조 내 일부 레이어를 양자 회로로 대체할 수 있도록 설계하였다. 양자 연산 부분의 구현은 IBM이 제공하는 Qiskit 라이브러리를 활용하였으며, 특히 실제 양자 하드웨어 대신 Qiskit Aer 시뮬레이터를 사용하여 NISQ 환경에서의 양자 연산 특성을 재현하였다. 양자 회로는 각 입력 이미지의 패치를 일정한 크기로 분할한 후, 해당 패치를 회전 게이트 기반의 파라메터라이즈드 회로(Parameterised Quantum Circuit)에 임베딩하는 방식으로 처리되었다. (학습 시 8코어 멀티스레딩 환경에서 1~3일 소요)

## Code Instruction

LeNet_test.py에서는 실험에서 비교할 LeNet기반 모델에 대한 함수를 가지고 있다. 양자컴퓨터를 사용하는 Quantum Model의 구현은 Theoric_QCNN_AlexNet.py와 QuantumLeNet_test.py에서 나누어서 담당하고 있다. 다음은 주요 코드에 대한 설명이다.

### Theoric_QCNN_AlexNet.py
GPU_BACKEND_OPTS, CPU_BACKEND_OPTS : AerSimulation에 넣을 각각의 CPU모드와 GPU 모드에 대한 argument이다.<br/>
USE_GPU : True or False로 설정하는지에 따라 AerSimulation에 넣을 모드가 자동으로 변경된다.<br/>
SHOTS : 평균값을 얻기 위해 측정하는 횟수인데, 회로 최적화 과정에서 시뮬레이션의 이점을 이용해 한번의 측정만으로 바로 평균값을 얻을 수 있도록 하였다. 따라서, 값을 바꿔도 코드 동작이 달라지지 않는다.
(function) QCTemplate : RX게이트를 사용한 인코딩을 진행 및 중앙 부분 파라미터에 의존하는 회로를 구현하고, 회로 최적화 및 AerSimulation으로 측정까지 실행하는 회로이다. 초기에는 여러 함수로 나눠 놓았지만, 최적화를 위해 한번만 실행되고 파라미터만 바꿔가면서 재사용하도록 하였다.<br/>
MAX_CIRCS_PER_RUN : 회로에서 멀티스레딩 시 한번에 회로 몇개씩 나누어 돌리지를 정하는 파라미터이다.<br/>
(function) QuanvBatchProbabilities : 여러 회로를 순회하면서 평균값을 모두 얻어내고 추가로 [0,1]의 feature map까지의 값으로 되돌린다. 한 번의 forward에서 회로를 모두 모아서 실행하도록 최적화 했디 때문에 이렇게 코드를 짜게 되었고, 이 함수 실행에서 시간이 가장 많이 소모된다.<br/>
(function) _make_binds : forward에서 회로를 모두 모아서 줌에 따라 파라미터와 회로를 각각 바인딩 하는 함수를 따로 만들었다.<br/>
(function) FasterQuanv3x3 : 모든 feature map을 한번에 모아 멀티스레딩으로 실행시켜 최대한 오버헤드를 줄이는 구조로 코드를 짰다. 이때, tensorflow에서 바로 사용할 수 있는 타입으로 바인딩 하여 최적화시켰다. 이 함수를 바로 QuantumLet_test.py에서 사용한다.

### QuantumLet_test.py
(class) Quanv3x3LayerClass : Tensorflow의 custom Layer 기능을 사용하기 위해 정의한 클래스이다. 여기서 Tensorflow에서 입출력 feature map의 크기를 명시해주는 매서드를 포함하고, 레이어를 호출했을 때 실행할 동작을 정의한다.<br/>
(function) _forward_run : forward 실행 시 Theorical_QCNN_AlexNet.py에서 정의한 함수를 사용한다.<br/>
(function) _spsa_grad : SPSA 방법론에 따라 총 3번의 forward를 파라미터값을 변경해 가며 호출하고, back propagation 할 값을 리턴한다.<br/>
(function) quantum_layer : 위에서 정의한 클래스 및 함수들을 모아서 실행하고, 파라미터 업데이트까지 진행한다. 이때, Quantum Layer는 backpropagation 시 앞에서 받은 레이어의 업데이트값을 그대로 받아 뒤로 넘겨 중간에 Quantum Layer를 삽입했을 때 Quantum에서 발생할 수 있는 노이즈를 최소화 하고자 하였다. (물론 다른 여러 방법도 시도해 봤으나 학습이 저조했다.)<br/>


### GPUtest.py

Qiskit Aer Simulation의 GPU 버전을 테스트 해볼 수 있다. 이 코드에서는 GPU 이득을 얻을 수 없어 제외하였다.

### Result.py

지금까지 얻은 실험데이터를 가지고 matplotlib을 통해 그래프를 만드는 코드이다.

### main.py

Ex~() 이름의 함수를 실행하여 각각의 모델들을 새롭게 학습시킬 수 있다. 학습 결과는 ‘실행한모델_Data’ 이름의 폴더에 저장된다. 또한, 이미 학습되어 있는 모델은 Test.py를 실행시켜 새로운 Test set에 대해서도 실행시킬 수 있다. 이때, 모델 파일의 경로 지정은 main 함수에서의 evaluate_and_plot 함수의 각각의 파라미터로 지정하고, 클래스 당 몇개씩 test set을 추출할 것인지 또한 파라미터로 지정시켜 결과를 확인할 수 있다.

## Demo

### 실험 결과

가정 1 검증
![alt text](/Result/test_accuracy_comparison.png)
교체 위치에 따른 Test set에 대한 epoch 별 Accuracy 비교

![alt text](/Result/test_loss_comparison.png)
교체 위치에 따른 Test set에 대한 epoch 별 Loss비교

![alt text](/Result/h1_param_count.png)
교체 위치에 따른 파라미터 개수 비교

![alt text](/Result/h2_test_accuracy_comparison.png)
교체 개수에 따른 Test set에 대한 epoch 별 Accuracy 비교

![alt text](/Result/h2_test_loss_comparison.png)
교체 개수에 따른 Test set에 대한 epoch 별 Accuracy 비교

![alt text](/Result/h2_param_count.png)
교체 개수에 따른 파라미터 수 비교

더 자세한 결과는 각 실험 결과에 대한 데이터 폴더 및 Result 폴더에서 확인할 수 있다.

## Conclusion and Future Work

가정 1. 네트워크의 첫 번째 합성곱 레이어를 Quantum 3×3 Convolution으로 치환했을 때, 모델의 정확도가 가장 높게 나타났으며, 수렴 속도 역시 다른 위치에 비해 가장 빠른 양상을 보였다. 이는 양자 회로가 네트워크 전반에 걸친 정보 흐름에 미치는 영향이 위치에 따라 크기 차이가 있으며, 특히 입력단의 저수준 특징 추출을 보강할 때 가장 효과적임을 시사한다.

 가정 2. 첫번째에서 세번째까지의 교체 결과에 대한 정확도 및 Loss는 거의 유사하게 나타났기 때문에 파라미터 수가 가장 적은 세 개의 레이어를 교체했을 때가 가장 효율적으로 나타났다. 반면 네 개의 레이어를 양자화하거나 출력단에 가까운 블록을 교체할 경우, 노이즈 및 barren plateau 문제가 누적되어 오히려 성능이 저하되는 경향을 보였다. 이러한 결과는 양자 레이어는 많을수록 좋은 것이 아니라, 언제, 어디에 사용하는지가 핵심이라는 통찰을 정량적으로 입증하는 결과이며, 제한된 큐비트 및 회로 깊이 조건을 갖는 NISQ 환경에서 실질적 성능을 달성하기 위한 최적 설계 기준을 제시한다는 점에서 큰 의미를 갖는다.

  이러한 연구 결과를 바탕으로, 여러 후속 연구 방향이 도출될 수 있다. 첫째, LeNet 외의 다양한 신경망 구조(AlexNet, MobileNet, ResNet 등)에 본 전략을 일반화할 수 있는지를 검증할 필요가 있다. 특히 각 CNN 구조에서는 여러 다양한 역할을 하는 레이어가 존재하는데, 이러한 복잡한 구조에서는 양자 레이어의 역할과 효과가 달라질 수 있으므로, 이에 대한 세부적인 분석이 요구된다.

 둘째, 실험 결과에서 나타난 얕은 회로와 소수의 큐비트 구성이 가장 효율적이라는 점은, 향후 양자 하드웨어 설계 단계에서 입력 전처리 전용 소형 큐비트 어레이를 탑재하는 등의 전략이 효과적일 수 있음을 시사한다. 이러한 하드웨어–알고리즘 공동 최적화를 통해 하이브리드 모델의 전력 소모와 열 발생을 줄이는 동시에, 실시간 추론 환경에서의 사용 가능성도 높일 수 있다.

 또한, NISQ 장치의 고유한 노이즈 특성에 대응하기 위해 노이즈를 완화시키는 Error Mitigation기법과 VQA에서 사용하는 변분 회로 최적화 전략을 통합함으로써, 출력단 근처에서도 성능 저하 없이 양자 레이어를 활용할 수 있는 가능성도 모색할 필요가 있다.

 마지막으로, 장기적으로는 수천 이상의 논리 큐비트 규모로 확장될 미래를 대비해, ImageNet 수준의 대규모 데이터셋과 수천만 개의 파라미터를 갖는 신경망 구조에서 양자화에 따른 파라미터 절감 효과, 에너지 효율성, 발열 저감 효과 등을 정량적으로 분석하는 확장 연구도 병행되어야 할 것이다.

 종합하면, 본 연구는 양자 레이어를 어디에, 얼마나 사용할 것인지에 대한 정량적 실험을 통해, 하이브리드 인공지능 구조의 설계에서 실질적인 설계 지침을 제공하였다. 이는 제한된 양자 자원을 효율적으로 활용하기 위한 기반을 마련함과 동시에, NISQ 환경에서도 의미 있는 성능 우위를 확보할 수 있는 실용적 Quantum AI 시스템 개발에 중요한 토대를 제공하였다는 의의가 있다.