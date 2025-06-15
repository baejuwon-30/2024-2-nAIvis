# Team-Info

| 항목 | 내용 |
|:--|:--|
| **과제명** | Elastic Weight Consolidation 기반 Continual Learning을 활용한 Deep Fingerprinting 성능 개선 |
| **팀 번호 / 팀 이름** | 11-nAIvis |
| **팀 구성원** | 배주원 (2271031): 리더, 전체적인 실험 및 논문작성 실행을 주도하며, 주요 사항을 최종 결정한다. <br> 신유진 (2271034): 팀원, 연구 및 실험을 진행하며 특히 보고서, 회의록 등 진행과정 기록을 주도한다. <br> 이서연 (2276217): 팀원, 연구 및 실험을 진행하며 특히 그래프, 표 등 결과 자료 제작을 주도한다. |
| **팀 지도교수** | 오세은 교수님 |
| **과제 분류** | 연구 과제 |
| **과제 키워드** | Catastrophic Forgetting, Continual Learning, Deep Fingerprinting |
| **과제 내용 요약** | Tor 네트워크 환경에서 Deep Fingerprinting(DF) 모델의 Catastrophic Forgetting 문제를 해결하고자, Elastic Weight Consolidation(EWC) 기반의 Continual Learning 기법을 적용한 새로운 DF 학습 프레임워크를 제안함. 다양한 Task 환경에서의 정확도 유지와 학습 효율성을 높이며, 실사용 가능한 보안 모델 개발을 목표로 함. |


# Project-Summary

| 항목 | 내용 |
|:--|:--|
| **문제 정의** | 기존 Deep Fingerprinting 모델은 정적인 데이터셋 기반으로 학습되어 시간에 따른 네트워크 트래픽 변화에 취약함. 이로 인해 새로운 Task 학습 시 기존 지식을 상실하는 Catastrophic Forgetting이 발생하며, 이는 실사용 환경에서 큰 성능 저하로 이어짐. |
| **기존 연구와의 비교** | 기존 연구들은 대부분 완전한 재학습 기반 혹은 정적 모델 구조를 사용함. 본 과제는 Elastic Weight Consolidation을 통해 지속적인 학습이 가능하며, 학습 비용을 줄이고 기존 Task 성능을 유지하는 장점을 가짐. |
| **제안 내용** | - 1D-CNN 기반 DF 모델에 EWC 적용<br>- Fisher Information을 기반으로 각 파라미터의 중요도를 계산<br>- 중요한 파라미터 변경을 억제하는 규제항 추가<br>- Closed-world / Open-world 환경 모두에서 실험 진행 |
| **기대효과 및 의의** | - 보안 분야에서 Continual Learning 적용 사례 제시<br>- 실제 Tor 환경에서도 학습 효율과 정확도를 동시에 확보<br>- Catastrophic Forgetting을 줄이는 데 성공한다면 향후 다양한 보안 시스템에 적용 가능성 확보<br>- 실시간 대응 가능성 증대로 학술적·산업적 기여 기대 |
| **주요 기능 리스트** | - Tor 트래픽 수집 및 전처리 자동화<br>- 1D-CNN 기반 DF 모델 구조 구현<br>- EWC 알고리즘 통합 및 FIM 계산<br>- Task 분할 및 실험 자동화 스크립트<br>- 정확도 및 망각 현상 분석<br>- 결과 시각화 및 성능 리포팅(논문) |

# Project-Design & Implementation

| 항목 | 내용 |
|:--|:--|
| **요구사항 정의** | **DF 모델이 현실 세계에 적용되었을 때 발생하는 성능 저하 문제 해결** : 기존 Deep Fingerprinting(DF) 모델은 각 웹사이트에 대한 네트워크 트래픽을 학습하여 사용자가 방문한 웹사이트를 예측하기 위해 설계되었다. 하지만 현실 세계에서 네트워크 트래픽 데이터는 새로운 페이지 출현, 페이지 내의 정보 변화 등 다양한 이유로 빠른 변화 양상을 보이게 된다. 이때 새롭게 변화된 Task를 학습할수록 기존 Task의 지식을 잃는(기존 태스크에 대한 분류 성능이 저하되는) 치명적인 문제인 Catastrophic Forgetting이 발생하기 때문에, 아무리 고정된 데이터셋에 대해 높은 성능을 기록한 모델을 현실 세계에 적용했을 때 그 성능이 유지되지 않는다. <br><br> **WF 분야에 지속 학습 기법 중 하나인 EWC 적용** : 위의 문제를 해결하기 위해 Continual Learning(지속 학습, CL) 기법을 도입하고자 한다. CL은 그 방법론에 따라 크게 Regularization-based, Memory Replay-based, Parameter Isolation-based로 분류되며, 이 중 Regularization-based 방법 중 하나인 Elastic Weight Consolidation(EWC)을 기반으로 한 Continual Learning 프레임워크를 도입할 것이다. 모델 파라미터의 업데이트 과정에 추가 항을 설정하여 과거 학습 정보의 보존과 새로운 정보의 학습을 동시에 가능하게 해야 한다.<br><br> **데이터셋을 분할하여 Open-World 환경 재현 및 EWC 성능 확인** : Open-world에서의 변화하는 데이터를 구현하기 위해 데이터를 적절히 분할하여 실험을 구성해야 하며, 반복 실험을 통해 주어진 task에 적합한 파라미터를 발견하고 추가한 손실 항이 기여하는 바를 명확히 밝혀야 한다. |
| **전체 시스템 구성** |![DFNet](./assets/model.png)<br>- 데이터 처리 모듈: datasets/ 디렉토리에는 Tor 트래픽 기반의 패킷 방향 시퀀스가 클래스별로 저장되어 있으며, 각 Task에 맞게 데이터를 불러와 학습용/평가용으로 분할함. 데이터는 NumPy 형식으로 전처리되며, 시퀀스의 길이는 10,000으로 고정됨. main.py에서 Task 순서에 따라 데이터셋을 자동으로 불러오고, 배치 크기 및 셔플 여부 등 학습 설정에 맞춰 전달함.<br><br>- 모델 정의 모듈: Model.py에는 Tor 트래픽의 패킷 방향 시퀀스를 입력으로 받아 웹사이트를 분류하는 1D-CNN 기반 DFNet 모델이 정의되어 있음. 이 모델은 5개의 Conv 블록으로 구성되며, 각 블록은 Conv1D → BatchNorm → ReLU/ELU → MaxPooling → Dropout 순으로 설계됨. 필터 수는 32~512로 점진적으로 증가하고, 커널 크기는 8로 고정되어 시계열의 지역 패턴을 효과적으로 추출함. 이후 Flatten()을 통해 출력을 1차원으로 변환하고, 두 개의 Dense 층(512 유닛)을 거쳐 Softmax로 최종 분류를 수행함. 전반적으로 과적합 방지를 위한 Dropout과 BatchNorm을 활용해 일반화 성능을 높인 구조임.<br><br>- EWC 적용 모듈ewc.py에서는 이전 Task 학습 완료 후, 모델 출력에 대한 그래디언트를 기반으로 각 파라미터의 Fisher Information을 근사 계산함. 선택된 학습 샘플에서 로그 가능도에 대한 기울기를 제곱하여 평균냄으로써 파라미터의 중요도를 추정하고, 이를 FIM으로 저장함. 이후 학습에서는 ewc_loss() 함수가 기존 CrossEntropy 손실에 EWC 정규화 항을 더하는 형태로 사용됨. 이 정규화 항은 각 파라미터의 변화량에 해당 파라미터의 중요도(Fisher 값)를 곱해 계산되며, Catastrophic Forgetting을 방지함. 전체 로직은 학습 루프 내에서 자동으로 호출되어 지속 학습을 지원함.<br><br>- 학습 및 평가 모듈: train.py에서는 주어진 데이터를 기반으로 Task 단위의 학습 루프를 수행함. 첫 Task는 CrossEntropy 손실 함수를 사용하여 기본 학습을 진행하고, 이후 Task부터는 EWC 손실 함수가 적용되어 이전 Task 지식의 보존을 유도함. 각 Task 학습 후에는 현재 Task에 대한 정확도와 함께, 이전 Task에 대한 정확도도 재평가하여 망각 지표(F)를 계산함. 학습에는 Adam Optimizer를 사용하며, 에폭 수, λ 값, 샘플 수 등의 하이퍼파라미터는 main.py에서 인자로 전달받아 유동적으로 설정됨.|
| **주요 엔진 및 기능 설계** | **1D-CNN 기반 DF 모델** <br> 입력으로 주어지는 Tor 트래픽의 방향 시퀀스 데이터(길이 10,000)는 Conv1D 계층을 통해 처리되며, 총 5개의 Convolutional 블록으로 구성됨. 각 블록은 Conv1D → BatchNormalization → Activation(ReLU 또는 ELU) 순으로 구성되고, 이어서 MaxPooling1D와 Dropout 계층이 적용되어 과적합을 방지함. 필터 수는 블록마다 32, 64, 128, 256, 512로 증가하며, 커널 크기는 모두 8로 고정되어 시퀀스의 지역적 패턴을 추출하는 데 최적화되어 있음. 이후 Flatten()을 통해 feature map을 1차원 벡터로 변환한 뒤, 두 개의 Fully Connected(Dense) 계층을 거쳐 마지막 Softmax 계층에서 웹사이트 클래스 확률을 출력함. 각 FC 계층은 512개의 유닛을 가지며, Batch Normalization과 Dropout이 추가되어 학습 안정성과 일반화 성능을 강화함. 전체 모델은 Tor 트래픽의 시계열 특성과 위치 정보를 효과적으로 포착할 수 있도록 설계되었으며, Deep Fingerprinting 환경에 최적화된 구조를 따름.<br><br>**EWC 손실 함수**: 기존의 Cross Entropy Loss 손실 함수에 다음과 같은 규제 항을 더함. Catastrophic Forgetting 완화<br>  $L_{\text{total}} = L_{\text{task}} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$<br><br>**Fisher Information Matrix 계산**:  Fisher Information Matrix(FIM)는 각 파라미터가 현재 Task의 예측 성능에 얼마나 기여하는지를 정량적으로 나타내는 지표로, Catastrophic Forgetting을 방지하는 핵심 요소로 사용됨. 본 프로젝트에서는 각 파라미터에 대한 손실 함수의 기울기(gradient)를 제곱하여 평균을 내는 방식으로 FIM을 근사함. 구체적으로는 선택된 학습 샘플들에 대해 로그 가능도(log-likelihood)의 그래디언트를 계산하고, 그 제곱값을 누적하여 평균냄으로써 각 파라미터별 중요도를 추정함. FIM 값이 클수록 해당 파라미터는 현재 Task에서 중요한 역할을 하며, 이후 Task 학습 시 손실 함수 내의 규제 항에서 더 큰 패널티가 부여되어 변화가 억제됨. 이러한 방식은 이전 Task에서 학습한 중요한 정보를 보존하면서도 새로운 Task에 대한 적응을 허용하여, 학습된 지식의 유지와 업데이트 간의 균형을 효과적으로 조절할 수 있게 함. <br><br>**Train Loop 구현**: 데이터 로딩부터 분할, 순차적인 학습과 평가까지의 파이프라인을 구현함. Incremental Learning이기 때문에 계획된 task 순서에 적합한 학습 데이터, 테스트 데이터를 구성하도록 인덱스를 조절하여 작성함. |
| **주요 기능의 구현** |**[1D-CNN 기반 DF 모델]** <br>**tensorflow 기반 구현** : 모델은 tensorflow에 내재된 특징 레이어를 설계에 따라 쌓는 방식으로 구현된다. DFNet이 처음 발표된 논문의 github repository 에서 모델을 가져와 사용하였다. <br><br> **모델 빌드** : build 메소드에 input 데이터 크기와 총 클래스의 개수를 입력하여 사용한다. 본 실험에서는 (10000, 1) 사이즈로 데이터를 입력하였으며 0 부터 94 까지 총 95개의 클래스를 사용하였다. <br><br><br>**[EWC 손실 함수]** <br><br>ce_loss = CetegoricalCrossentropy<br>ewc_loss = compute_ewc_penalty<br>return ce_loss + ewc loss<br><br> EWC는 기본 CE Loss에 규제항으로서 추가되어 작동한다. 중요한 모델의 구성요소가 크게 변화할수록 추가적인 로스를 부여하는 것이다. <br><br>**compute_ewc_penalty** <br>loss += tf.reduce_sum(F * (c – o)^2)) <br> F는 해당 weight에 계산된 fisher information 값, c(current)는 현재 weight의 값, o(old)는 이전 태스크 학습이 완료되었을 때 weight의 값이다. 각 파라미터별로 페널티를 계산하여 중요한 파라미터의 변화에는 큰 페널티를, 중요하지 않은 파라미터의 변화에는 작은 페널티를 부여한다. <br><br> **return loss * (lamb / 2)** : lamb(lambda)는 EWC의 강도를 의미한다. 람다가 0이면 Loss에 EWC penalty 값이 적용되지 않고, 큰 값이 입력될수록 최종 로스에서 EWC penalty가 차지하는 비율이 커지므로 모델은 파라미터 변화에 더욱 보수적으로 학습된다. <br><br>**최종적으로 ce_loss + ewc loss = ce_loss + ewc_penalty * (lamb / 2) 형태로 수식을 구현한다.** <br><br><br>**[Fisher Information Matrix 계산]**<br>모델이 주어진 데이터에 대해 학습 한 후, 문제 해결에 중요한 역할을 하는 요소를 파악하고 그 정도를 정량화 하는 과정이다. <br><br>**compute_fisher_matrix(model, data, num_sample, epsilon)** <br>우선 함수는, 모델이 가진 학습 가능한(변화 가능한) 전체 weights 크기의 빈 리스트를 만든다. 이후 주어진 데이터에서 num_sample 만큼의 데이터를 무작위 선택하여 fisher information 계산에 활용할 것이다. <br><br>output = model(x, training=False) : 선택한 데이터를 하나씩 모델에 통과시켜(forward) 결과를 저장한다. 이때 모델의 마지막 레이어가 softmax이기 때문에 출력은 각 class일 확률이다. <br><br>output = tf.clip_by_value(output, epsilon, 1.0) : 이후 output에 로그를 취할 때, 0 값이 입력된다면 오버플로우가 발생한다. 이 오버플로우를 방지하지 위해 아주 작은 값인 epsilon 을 기준으로 값을 절단한다. <br><br>gradients = tape.gradient(log_likelihood, weights) : output에 로그를 취한 값과 weight로 그래디언트를 계산하여, 각 weight가 결과에 어느 정도의 영향을 주는지 계산한다. 이 영향력이 클수록 중요한 파라미터로 판단할 수 있다. <br>variance += tf.squre(gradients) <br><br> fisher_matrix = [ v / num_sample for v in variance ] : 여러 개의 데이터 각각의 그래디언트를 제곱 누적한 후, 데이터의 총 개수로 그 값을 나누어 평균 값을 fisher matrix에 저장한다. 데이터 하나가 이상치였을 때의 위험을 줄이고, 더욱 일반적인 중요도를 계산할 수 있도록 한다. <br><br><br>**[ Train Loop 구현]**<br><br>**파라미터의 구성**<br>- model : 학습시킬 모델을 입력한다. 모델을 train_loop에 들어가기 전에 빌드 되어야 하며, 본 연구에서는 DFNet이 함수에 전달된다. <br>- OPTIMIZER : 사용자가 설정한 OPTIMIZER를 입력한다. <br>- MAX_LABEL : 데이터가 가지고 있는 클래스의 총 개수를 나타낸다. Incremental Learning 진행 시 반복문을 빠져나오는 과정과 데이터 전처리 과정에서 사용한다. <br>- data : 사용할 데이터를 입력한다. 함수 내에서 학습/테스트 데이터 분리와 feature/label 분리가 일어나기 때문에, 데이터를 분할하여 입력할 필요는 없다. <br>- test_size : 함수 안에서 테스트 데이터를 분리할 때, 데이터 비율을 설정한다. <br>- first_task : 최초 학습 시 사용할 데이터셋의 분할 기준을 입력한다. 49를 입력한다면 0~49 범위의 50개의 클래스가 최초 학습 데이터로 선택된다. <br>- inc_task : Incremental Learning 시 증가하는 데이터의 크기를 결정한다. 5를 입력한다면 각 증가 스텝마다 5개의 데이터가 추가된다. <br>- first_epochs, inc_epochs : 각각 최초 학습 과정의 epoch 수와 추가 학습 과정의 epoch 수를 의미한다. <br>- lamb : EWC의 강도를 설정한다. 0을 입력하면 EWC가 적용되지 않으며, 값이 커질수록 강한 규제가 적용된다. <br>- num_sample : Fisher Information Matrix 계산을 위해 사용되는 데이터 샘플의 수를 의미한다. 값이 커질수록 계산량이 늘어나고 정확성이 증가한다. 주어진 데이터의 수를 초과 할 수 없고, 초과 값을 입력했을 때는 주어진 데이터 샘플을 모두 사용한다.  <br><br>**학습 데이터의 분할**<br>본 프로젝트는 95개의 Tor 웹사이트를 두 Task로 분할한 Closed-world 시나리오에 따라 실험을 수행함. Train_loop 함수에 입력 변수로 전체 데이터와 분할 기준을 받아 데이터의 레이블을 기준으로 데이터를 이분할 한다. 이때 루프 안에서 간단한 코드로 이 작업을 수행하기 위해 utils.py 파일에 split_by_label 함수를 구현하였다. split_by_label에서는 pandas 모듈의 데이터 형식인 dataframe의 기본 연산을 활용하며, 파라미터로 입력된 inc_task 값을 활용하여 3 task 이상으로 설계된 Incremental Learning에서 적절한 데이터를 선택하는 모듈로도 사용된다. <br><br>**테스트 데이터의 누적**<br> 모델이 처음 학습한 task를 어느 정도 망각하였는지 파악하기 위해서, 두번째 태스크를 학습시킨 상태에서 처음 학습한 task의 데이터로 구성된 test dataset을 활용하여 모델을 평가해야 한다. 따라서 처음 학습 task 과정에서 test 데이터를 보관하며, 다음 학습 후에 model.evauation 함수를 사용하여 보관한 데이터셋에 대해 평가를 진행한다. <br> 다만 본 연구는 2 task 문제만을 다루고 있지만, 이후 3 task 이상의 incremental learning의 원활한 진행을 위해 loop 안에 데이터를 계속해서 누적할 수 있도록 작성하였다. 학습과 평가가 끝난 뒤 누적 데이터셋을 업데이트 하는 방식을 사용하여, n번째 학습을 후 n번째 데이터에 대한 평가와 n-1번째 데이터에 대한 평가가 모두 수행될 수 있도록 알고리즘을 작성하였다. <br><br>**EWC 기반 지속 학습 구현** : ewc.py를 통해 Fisher Information Matrix(FIM)를 계산한 후, 이전 Task에서 중요한 파라미터 변화에 패널티를 부과하는 손실 함수를 정의한다. 이를 통해 기존 Task의 성능을 유지하면서 새로운 Task를 학습할 수 있다.<br><br>**실험 종류 및 설정**<br>① Baseline 비교 실험: EWC 없이 순차적으로 학습하는 Non-Baseline, 모든 데이터를 한 번에 학습하는 Joint Learning, 그리고 EWC 기반 학습 성능을 비교함.<br>② 클래스 비율 변화 실험: Task 1과 Task 2에 할당되는 클래스 수를 90:5, 70:25, 50:45로 조정하여 클래스 비중이 Catastrophic Forgetting 완화에 미치는 영향을 실험함.<br>③ 람다(λ) 값 변화 실험: 정규화 계수 λ 값을 1, 5, 100, 1000으로 변경하여 EWC 손실 항의 영향력을 조정하고, 기존 정보 보존과 새로운 학습 간의 trade-off를 분석함.<br>④ 에폭 비율 변화 실험: Task 1과 Task 2의 epoch 비율을 20:20, 50:20, 100:20 등으로 달리하여 학습량의 차이가 망각 현상에 어떤 영향을 주는지 평가함.<br><br>- 성능 평가 방식: 각 Task 학습이 완료된 후, Task 1과 Task 2 데이터셋 모두에 대해 평가를 수행하며, 다음의 지표를 기준으로 실험 결과를 비교함:<br>∙ Task별 정확도 (T1, T2 Accuracy)<br>∙ 평균 정확도 (Average Accuracy)|
| **기타** | - 학습 환경: TensorFlow 기반으로 Google Colab 또는 로컬 GPU 환경에서 실험 수행<br>- 재현성 확보: 모든 실험에는 동일한 random seed를 고정하여 일관된 결과를 생성함<br>- 코드 관리 및 협업: GitHub 저장소(hineugene/continual-learning-DF)를 통해 코드 버전 관리 및 협업을 지원하며, 각 파일에는 명확한 주석과 디렉토리 구조 설명이 포함되어 있음 |

## Evaluation (평가)

### 평가 항목

1. 정확도 유지 능력 (Accuracy Retention)
2. 망각 완화 능력 (Catastrophic Forgetting Mitigation)
3. 학습 효율성 (Learning Efficiency)

### 평가 기준

* 정확도 유지: 각 Task별 정확도의 평균이 90% 이상일 경우 우수
* 망각 완화: 이전 Task에 대한 정확도 감소가 20% 이하로 유지될 경우 우수
* 학습 효율성: 신규 Task 학습 시 이전 Task 정확도 유지와 신규 Task 학습 정확도 상승 비율이 균형을 이루었을 때 우수

### 평가 방식

* 각 Task 학습 후 정확도 측정 및 평균값 계산
* Task 간 정확도 변화량을 분석하여 Catastrophic Forgetting 평가
* Fisher Information Matrix를 활용하여 파라미터 중요도 분석 및 EWC 효과성 평가

### 평가 내용 및 결과

| 평가 항목                 | 실험 결과 요약                                                        | 정량적 지표                                    |
| :-------------------- | :-------------------------------------------------------------- | :---------------------------------------- |
| **정확도 유지 능력**         | EWC 모델은 Task 1 정확도를 평균 90% 이상 유지함                               | Task 1 Accuracy: 92.1% 이상                 |
| **망각 완화 능력**          | Task 2 학습 이후에도 Task 1 정확도 감소율이 20% 미만                           | Accuracy Drop < 20%                       |
| **학습 효율성**            | Lambda 값과 Epoch 비율 조절을 통해 기존 정보 보존과 신규 학습 간 균형 확보               | λ=100에서 평균 정확도 89.7% 유지                   |
| **Baseline 대비 성능 향상** | EWC는 Non-Baseline 대비 평균 정확도 5% 이상 향상                            | Non: 83.9%, EWC: 89.8%                    |
| **Joint 모델과의 비교**     | Joint 학습 대비 Catastrophic Forgetting은 존재하나, 실제 환경 적용 측면에서 효율성 확보 | Joint: 97.5%, EWC: 89.8% (망각 대비 성능 유지 우수) |

각 실험은 동일한 조건에서 3회 반복되었으며, 평균값 기준으로 작성함.

* **정확도 유지 능력**: EWC를 적용한 모델은 Non-Baseline 대비 Task 1 정확도가 평균 90% 이상 유지되어 우수한 성능을 보임
* **망각 완화 능력**: EWC 적용 시 Task 2 학습 후 Task 1의 정확도 감소율이 20% 미만으로 유지되어 Catastrophic Forgetting이 효과적으로 완화됨
* **학습 효율성**: Lambda 값과 Epoch 비율을 최적화하여 기존 지식 유지와 신규 학습 간 균형을 잘 유지하며, 특히 첫 번째 Task의 학습 Epoch을 증가시킬수록 장기적 성능 유지가 안정적인 것을 확인함 

## 결론 및 기대효과

본 연구는 EWC 기반 Continual Learning을 DF 모델에 적용하여 Catastrophic Forgetting 완화 및 성능 개선을 실증적으로 입증하였으며, 이는 정적인 학습 방식 대비 발전된 접근으로 향후 다양한 보안 시스템의 실시간 대응 및 적용 가능성을 높일 것으로 기대한다.

* **DF모델 취약점 보완**: EWC(Elastic Weight Consolidation) 기법을 적용하여, DFNet에서 반복 학습 시 발생하는 성능 저하(Catastrophic Forgetting) 문제를 완화하였다. 실제로 추가 학습을 진행한 이후에도, 이전 데이터에 대한 예측 성능이 잘 보존되는 것을 확인하였다. 이는 DFNet이 지속적인 학습 환경에서도 기존 학습 내용을 안정적으로 유지할 수 있음을 보여준다.
* **데이터 변화 시나리오에서의 DF모델 성능 확인**: 전체 데이터셋을 클래스 단위의 task로 분할하고 각 Task를 순차적으로 학습하는 실험을 진행하였다. 이 과정에서 EWC를 적용한 경우, 적용하지 않았을 때의 결과에 비해 성능 저하 정도의 폭이 유의미하게 감소함을 확인할 수 있었다. 이를 통해 DF모델이 데이터의 시간에 따른 점진적 변화에 적응하면서도 이전 Task에 대한 정보를 보다 잘 보존할 수 있음을 확인하였다.
* **DF모델에서의 CL기법 효과 확인**: EWC의 Lambda 값과 각 Task의 Epoch 비율을 조정해가며 실험을 진행한 결과, 기존 지식을 유지하면서도 새로운 정보에 대한 학습 성능을 확보하는 데 있어 균형이 중요한 요소로 작용함을 확인하였다. 특히 첫 번째 Task의 학습 Epoch 수를 늘릴수록, 전체적인 성능 유지가 더 안정적이라는 점도 발견되었다. 이는 모델이 초기에 형성한 표현이 이후 Task에서도 중요한 역할을 하기 때문으로 해석된다. EWC 외에도 다양한 지속 학습 기법이 존재하며, 본 실험 결과를 바탕으로 Replay 기반 방법이나 Parameter Isolation 계열 기법들도 DFNet에서 효과를 보일 가능성이 높다고 판단된다. 본 연구에서는 단 한 번의 추가 학습에서 기존 Task의 성능을 상당 수준 유지할 수 있었으며, 반복적인 추가 학습 환경에서도 성능 보존 효과를 기대할 수 있을것이다.

