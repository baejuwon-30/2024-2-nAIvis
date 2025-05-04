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

# ⚙️ Project-Design & Implementation

| 항목 | 내용 |
|:--|:--|
| **요구사항 정의** | 본 프로젝트는 Deep Fingerprinting 모델이 직면한 Catastrophic Forgetting 문제를 해결하기 위한 지속 학습 프레임워크를 설계하는 것을 목표로 한다. 이를 위해 Elastic Weight Consolidation(EWC)을 통합하여 기존 학습된 Task의 파라미터를 보존하면서도 새로운 Task에 대한 학습이 가능해야 한다. 또한, 실험 시나리오로는 총 95개 클래스를 사용하는 Closed-world와 50+45 Task로 구성된 Open-world 환경을 모두 지원해야 하며, 학습된 모델의 성능을 정량적으로 평가할 수 있는 지표(정확도, 망각률 등)와 시각화 도구가 요구된다. |
| **전체 시스템 구성** | 프로젝트 전체는 크게 다섯 개의 구성 요소로 이루어진다. (1) `datasets/` 디렉토리에서는 실험에 사용할 Tor 트래픽 데이터를 분할 및 저장하며, Task별 데이터셋 구성을 지원한다. (2) `Model.py` 파일에서는 1D-CNN 기반의 DF 모델을 정의하고, 입력 시퀀스를 처리할 네트워크 구조를 설계한다. (3) `ewc.py`는 기존 학습된 파라미터의 중요도를 평가하는 FIM(Fisher Information Matrix)을 계산하고, 손실 함수에 이를 반영하는 EWC 손실 항을 구현한다. (4) `train.py`는 각 Task에 대한 학습 루프를 정의하고 정확도 등의 성능 지표를 측정한다. (5) `main.py`는 전체 실험 흐름을 제어하며, Task 전환, 모델 저장/불러오기, 시각화 여부 등 파라미터를 설정한다. |
| **주요 엔진 및 기능 설계** | 본 시스템의 핵심은 1D-CNN 기반 DF 모델과 EWC 손실 함수를 결합하는 구조이다. 모델은 Tor 패킷 방향 시퀀스 데이터를 Embedding Layer를 통해 저차원 벡터로 매핑한 뒤, Conv1D 층에서 특징을 추출하고, Global Average Pooling을 거쳐 Dense-Softmax 계층에서 웹사이트를 분류한다. EWC 손실 함수는 $L_{\text{total}} = L_{\text{task}} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$ 형태로 구현되며, 여기서 $F_i$는 각 파라미터의 중요도를 나타내는 FIM 값, $\theta_i^*$는 이전 Task에서의 최적 파라미터를 의미한다. 이 규제항을 통해 중요한 파라미터의 변화는 억제되고, 덜 중요한 파라미터 위주로 새로운 Task가 학습된다. |
| **주요 기능의 구현** | 학습 자동화를 위해 `main.py`에서 Task 간 데이터를 자동 분할하고, 각 Task에 대해 `train.py`가 반복적으로 호출된다. 학습은 처음 Task에서 더 긴 epoch(70)을 부여하여 충분히 학습시키고, 이후 Task들은 상대적으로 짧은 epoch(20)으로 빠르게 적응할 수 있도록 설계하였다. 실험은 Closed-world와 Open-world 두 가지로 수행되며, Open-world에서는 모델이 이전에 학습하지 않은 클래스에 대해 일반화되는 능력까지 평가된다. 성능 평가는 Task별 정확도, 평균 정확도, Forgetting 지표(F), 최종 정확도 등 다양한 측면에서 이뤄지며, 결과는 CSV로 저장되어 분석이 가능하다. 또한 Matplotlib을 활용하여 학습 곡선 및 망각률 그래프를 자동으로 시각화하고 저장한다. |
| **기타** | 본 실험은 Google Colab 및 로컬 GPU 환경에서 TensorFlow 기반으로 수행되며, reproducibility 확보를 위해 모든 학습 과정에 대해 random seed를 고정하였다. 실험 코드, 데이터 로딩, 모델 정의, 평가 함수 등은 GitHub 저장소 [hineugene/continual-learning-DF](https://github.com/hineugene/continual-learning-DF)에 공개되어 있으며, 주석과 디렉토리 구조를 통해 팀원 간 협업이 원활하게 이루어지도록 설계되어 있다. 실험 결과는 별도의 정리 리포트를 통해 요약되며, 이 보고서에도 성능 비교표와 학습/망각 시각화 자료가 포함될 예정이다. |
