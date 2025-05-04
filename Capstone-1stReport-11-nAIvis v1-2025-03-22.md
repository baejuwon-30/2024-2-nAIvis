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
| **전체 시스템 구성** | - 데이터 처리 모듈: datasets/ 디렉토리에 저장된 Tor 트래픽 데이터를 Task 단위로 불러오고 전처리하는 기능을 수행<br>- 모델 정의 모듈: Model.py에서 1D-CNN 구조를 정의하며, 입력 시퀀스를 Embedding하고 Convolution 및 Pooling을 통해 특징을 추출한 뒤 Softmax 분류를 수행<br>- EWC 적용 모듈: ewc.py에서 Fisher Information Matrix(FIM)를 계산하고, 이를 바탕으로 손실 함수에 규제 항을 추가하여 파라미터 보존을 유도<br>- 학습 및 평가 모듈: train.py에서 학습 루프를 정의하고, 정확도 및 망각률 등 다양한 지표를 측정<br>- 실험 통제 모듈: main.py에서 전체 파이프라인을 제어하며, Task 전환, 하이퍼파라미터 설정, 결과 저장 및 시각화를 수행 |
| **주요 엔진 및 기능 설계** | - 1D-CNN 기반 DF 모델: 입력으로 주어지는 Tor 트래픽의 방향 시퀀스 데이터(길이 10,000)는 먼저 Embedding Layer를 통해 고차원 벡터로 변환됨. 이후 Conv1D 층에서 필터 수 128, 커널 크기 8을 사용하여 지역적인 특징을 추출함. 활성화 함수로는 ReLU를 사용하며, Global Average Pooling을 통해 시퀀스 차원을 축소함. 마지막으로 Fully Connected(Dense) 계층을 거쳐 Softmax를 통해 웹사이트 클래스를 예측함. 전체 구조는 간결하지만 Tor 트래픽의 시퀀스적 특성을 효과적으로 포착하도록 설계됨.<br>- **EWC 손실 함수**: 기존 손실 함수에 다음과 같은 규제 항을 추가하여 Catastrophic Forgetting 완화<br>  $L_{\text{total}} = L_{\text{task}} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$<br>- **FIM 계산**: 학습된 모델의 파라미터 각각이 현재 Task에서 얼마나 중요한지를 나타내는 값으로, 각 파라미터의 기울기 제곱 평균을 통해 근사적으로 계산됨. FIM 값이 클수록 해당 파라미터는 현재 Task에서 중요한 역할을 하며, 이후 Task 학습 시 손실 함수에서 더 큰 가중치를 갖게 되어 변경을 억제하게 됨. 이를 통해 학습된 지식의 보존과 새로운 정보의 통합 간 균형을 유지함. |
| **주요 기능의 구현** | - Task 기반 학습 구조: main.py에서 Task 단위로 데이터셋을 분할하고, Task별 학습 루프를 자동으로 실행함<br>- 점진적 학습 전략: 초기 Task에는 70 epoch, 이후 Task에는 20 epoch으로 설정하여 점진적으로 새로운 지식을 통합함<br>- Closed-world / Open-world 대응: Task split 전략에 따라 두 가지 시나리오를 모두 처리할 수 있도록 데이터 및 모델 학습을 설계함<br>- 성능 평가 지표: Accuracy, Average Accuracy, Forgetting Rate(F), Final Accuracy 등의 지표를 각 Task마다 기록하며 비교 분석함<br>- 시각화 및 결과 저장: 정확도 및 망각률 그래프는 Matplotlib을 사용하여 자동 생성되며, 각 실험 결과는 CSV 형태로 저장되어 추후 비교 분석이 용이함 |
| **기타** | - 학습 환경: TensorFlow 기반으로 Google Colab 또는 로컬 GPU 환경에서 실험 수행<br>- 재현성 확보: 모든 실험에는 동일한 random seed를 고정하여 일관된 결과를 생성함<br>- 코드 관리 및 협업: GitHub 저장소(hineugene/continual-learning-DF)를 통해 코드 버전 관리 및 협업을 지원하며, 각 파일에는 명확한 주석과 디렉토리 구조 설명이 포함되어 있음 |
