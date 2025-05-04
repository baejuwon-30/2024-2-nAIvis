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
| **요구사항 정의** | - 기존 Deep Fingerprinting(DF) 모델의 Catastrophic Forgetting 문제 해결<br>- Elastic Weight Consolidation(EWC) 기반 Continual Learning 적용<br>- Closed-world 및 Open-world 시나리오 모두 지원<br>- 실험 자동화 및 결과 시각화 도구 포함 |
| **전체 시스템 구성** | - **데이터셋 구성**: `datasets/` 디렉토리 내에 Tor 트래픽 데이터를 저장 및 관리<br>- **모델 정의**: `Model.py`에서 1D-CNN 기반 DF 모델 구조 정의<br>- **EWC 구현**: `ewc.py`에서 Fisher Information Matrix(FIM) 계산 및 EWC 손실 항 추가 구현<br>- **학습 및 평가**: `train.py`에서 학습 루프 및 평가 지표 계산 수행<br>- **실험 제어**: `main.py`에서 실험 파라미터 설정 및 전체 파이프라인 제어 |
| **주요 엔진 및 기능 설계** | - **1D-CNN 기반 DF 모델**: 입력 시퀀스(길이 10,000)를 Embedding Layer를 통해 처리한 후, Conv1D → Pooling → Softmax 구조로 구성<br>- **EWC 손실 함수**: 기존 손실 함수에 다음과 같은 규제 항을 추가하여 Catastrophic Forgetting 완화<br>  $L_{\text{total}} = L_{\text{task}} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$<br>- **FIM 계산**: 각 파라미터의 중요도를 평가하여 EWC 손실 항에 반영 |
| **주요 기능의 구현** | - **Task 분할 및 학습 자동화**: `main.py`에서 Task별 데이터 분할 및 학습 루프 자동화 구현<br>- **학습 전략**: 첫 번째 Task는 70 epoch, 이후 Task는 20 epoch으로 설정하여 점진적 학습 수행<br>- **시나리오 지원**: Closed-world 및 Open-world 환경 모두에서 실험 가능<br>- **성능 평가 지표**: Task별 정확도(Accuracy), 평균 정확도(Average Accuracy), Catastrophic Forgetting 지표(F), 최종 정확도(Final Accuracy) 등 다양한 지표 계산<br>- **결과 시각화 및 저장**: Matplotlib을 이용하여 정확도 및 망각 현상 관련 그래프 자동 생성, 결과 데이터는 CSV 형태로 로그 저장하여 지속적 성능 추적 가능 |
| **기타** | - **실험 환경**: GPU 환경에서 Python 및 TensorFlow 기반으로 학습 수행<br>- **재현성 확보**: 실험 결과의 재현성을 위해 random seed 고정<br>- **코드 관리**: GitHub 저장소를 통해 코드 버전 관리 및 협업 지원 |
