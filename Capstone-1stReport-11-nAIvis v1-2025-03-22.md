# Team-Info

| 항목 | 내용 |
|:--|:--|
| **과제명** | Elastic Weight Consolidation 기반 Continual Learning을 활용한 Deep Fingerprinting 성능 개선 |
| **팀 번호 / 팀 이름** | 11-nAIvis |
| **팀 구성원** | 배주원 (리더), 신유진, 이서연 |
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
| **주요 기능 리스트** | - Tor 트래픽 수집 및 전처리 자동화<br>- 1D-CNN 기반 DF 모델 구조 구현<br>- EWC 알고리즘 통합 및 FIM 계산<br>- Task 분할 및 실험 자동화 스크립트<br>- 정확도 및 망각 현상 분석 도구<br>- 결과 시각화 및 성능 리포팅 시스템 |

# Project-Design & Implementation

| 항목 | 내용 |
|:--|:--|
| **요구사항 정의** | - 기존 Deep Fingerprinting(DF) 모델이 시간에 따라 발생하는 트래픽 패턴의 변화에 민감하여, 새로운 Task를 학습할수록 기존 Task의 지식을 잃는 Catastrophic Forgetting 문제가 존재함<br>- 이를 해결하기 위해 Elastic Weight Consolidation(EWC)을 기반으로 한 Continual Learning 프레임워크를 도입하여, 과거 학습 정보의 보존과 새로운 정보의 학습을 동시에 가능하게 해야 함<br>- Closed-world(95개 클래스) 및 Open-world(50+45 Task) 환경 모두에서 학습이 가능해야 하며, 반복 실험을 통한 정량적 성능 분석과 결과 시각화까지 포함된 실험 자동화 구조가 요구됨 |
| **전체 시스템 구성** |- 데이터 처리 모듈: datasets/ 디렉토리에는 Tor 트래픽 기반의 패킷 방향 시퀀스가 클래스별로 저장되어 있으며, 각 Task에 맞게 데이터를 불러와 학습용/평가용으로 분할함. 데이터는 NumPy 형식으로 전처리되며, 시퀀스의 길이는 10,000으로 고정됨. main.py에서 Task 순서에 따라 데이터셋을 자동으로 불러오고, 배치 크기 및 셔플 여부 등 학습 설정에 맞춰 전달함.<br><br>- 모델 정의 모듈: Model.py에는 1D-CNN 기반의 Deep Fingerprinting 분류기가 정의됨. 입력 시퀀스를 Embedding Layer에서 고차원 벡터로 변환하고, 이후 Conv1D 계층(128 filters, kernel size=8, activation=ReLU)을 통해 로컬 특징을 추출함. Global Average Pooling 계층으로 시퀀스 축소 후, Dense → Softmax 계층에서 최종 클래스 확률 분포를 출력함. 이 구조는 패킷 시퀀스의 시계열적 특성과 지역적 변화를 효과적으로 포착하도록 설계됨.<br><br>- EWC 적용 모듈: ewc.py에서는 이전 Task 학습이 완료된 후, 해당 Task에서의 손실 함수의 그래디언트 정보를 바탕으로 각 파라미터의 Fisher Information을 근사적으로 계산함. 계산된 FIM과 파라미터 저장값(θ*)는 이후 Task 학습 시 규제 항으로 손실 함수에 반영되며, Catastrophic Forgetting을 억제함. 이 모듈은 파라미터별 중요도를 저장하고, 학습 루프에서 loss 계산 시 자동으로 호출됨.<br><br>- 학습 및 평가 모듈: train.py에서는 주어진 데이터에 대해 학습 루프를 수행하고, 각 Task가 끝날 때마다 모델을 평가함. 학습에는 Adam Optimizer 및 CrossEntropy Loss를 사용하며, 학습 정확도 외에도 이전 Task에 대한 정확도 변화를 기록하여 망각 지표(F)를 계산함. 반복 학습 시 학습률, 에폭 수, λ 값 등은 main.py에서 인자로 전달받아 유연하게 설정됨.<br><br>- 실험 통제 모듈: main.py는 전체 실험의 흐름을 관리하는 핵심 제어 모듈로, Task 간 전환, 하이퍼파라미터 설정, 모델 초기화 및 저장 등의 기능을 포함함. 실험 조건에 따라 Non-Baseline / EWC / Joint Line 구조로 학습 모드를 지정할 수 있으며, 실험 결과는 콘솔 출력을 통해 실시간 확인 가능함. 시각화 및 CSV 저장 기능은 포함되어 있지 않으며, 결과는 수기로 기록됨. |
| **주요 엔진 및 기능 설계** | - 1D-CNN 기반 DF 모델: 입력으로 주어지는 Tor 트래픽의 방향 시퀀스 데이터(길이 10,000)는 먼저 Embedding Layer를 통해 고차원 벡터로 변환됨. 이후 Conv1D 층에서 필터 수 128, 커널 크기 8을 사용하여 지역적인 특징을 추출함. 활성화 함수로는 ReLU를 사용하며, Global Average Pooling을 통해 시퀀스 차원을 축소함. 마지막으로 Fully Connected(Dense) 계층을 거쳐 Softmax를 통해 웹사이트 클래스를 예측함. 전체 구조는 간결하지만 Tor 트래픽의 시퀀스적 특성을 효과적으로 포착하도록 설계됨.<br>- **EWC 손실 함수**: 기존 손실 함수에 다음과 같은 규제 항을 추가하여 Catastrophic Forgetting 완화<br>  $L_{\text{total}} = L_{\text{task}} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$<br>- **FIM 계산**: 학습된 모델의 파라미터 각각이 현재 Task에서 얼마나 중요한지를 나타내는 값으로, 각 파라미터의 기울기 제곱 평균을 통해 근사적으로 계산됨. FIM 값이 클수록 해당 파라미터는 현재 Task에서 중요한 역할을 하며, 이후 Task 학습 시 손실 함수에서 더 큰 가중치를 갖게 되어 변경을 억제하게 됨. 이를 통해 학습된 지식의 보존과 새로운 정보의 통합 간 균형을 유지함. |
| **주요 기능의 구현** |- Task 기반 학습 구조: 본 프로젝트는 95개의 Tor 웹사이트를 두 Task로 분할한 Closed-world 시나리오에 따라 실험을 수행함. main.py에서 Task 단위로 데이터를 분할하고, 각 Task에 대해 순차적으로 학습을 수행함으로써 모델이 점진적으로 새로운 클래스를 학습할 수 있도록 구성함. 각 Task 학습 시에는 train.py 내 학습 루프를 호출하여 독립적인 모델 업데이트가 이루어짐.<br><br>- 점진적 학습 전략 적용: 첫 번째 Task 학습 시에는 70 epoch을 적용하여 초기 정보 학습을 충분히 수행하고, 두 번째 Task에서는 20 epoch으로 설정하여 빠른 적응이 가능하도록 구성함. 이 설정은 초기 학습 안정성과 새로운 정보 반영의 균형을 고려한 전략임.<br><br>- EWC 기반 지속 학습 구현: ewc.py를 통해 Fisher Information Matrix(FIM)를 계산한 후, 이전 Task에서 중요한 파라미터 변화에 패널티를 부과하는 손실 함수를 정의함. 이를 통해 기존 Task의 성능을 유지하면서 새로운 Task를 학습함.<br><br>- 실험 종류 및 설정:<br>① Baseline 비교 실험: EWC 없이 순차적으로 학습하는 Non-Baseline, 모든 데이터를 한 번에 학습하는 Joint Learning, 그리고 EWC 기반 학습 성능을 비교함.<br>② 클래스 비율 변화 실험: Task 1과 Task 2에 할당되는 클래스 수를 90:5, 70:25, 50:45로 조정하여 클래스 비중이 Catastrophic Forgetting 완화에 미치는 영향을 실험함.<br>③ 람다(λ) 값 변화 실험: 정규화 계수 λ 값을 1, 5, 100, 1000으로 변경하여 EWC 손실 항의 영향력을 조정하고, 기존 정보 보존과 새로운 학습 간의 trade-off를 분석함.<br>④ 에폭 비율 변화 실험: Task 1과 Task 2의 epoch 비율을 20:20, 50:20, 100:20 등으로 달리하여 학습량의 차이가 망각 현상에 어떤 영향을 주는지 평가함.<br><br>- 성능 평가 방식: 각 Task 학습이 완료된 후, Task 1과 Task 2 데이터셋 모두에 대해 평가를 수행하며, 다음의 지표를 기준으로 실험 결과를 비교함:<br>∙ Task별 정확도 (T1, T2 Accuracy)<br>∙ 평균 정확도 (Average Accuracy)|
| **기타** | - 학습 환경: TensorFlow 기반으로 Google Colab 또는 로컬 GPU 환경에서 실험 수행<br>- 재현성 확보: 모든 실험에는 동일한 random seed를 고정하여 일관된 결과를 생성함<br>- 코드 관리 및 협업: GitHub 저장소(hineugene/continual-learning-DF)를 통해 코드 버전 관리 및 협업을 지원하며, 각 파일에는 명확한 주석과 디렉토리 구조 설명이 포함되어 있음 |
