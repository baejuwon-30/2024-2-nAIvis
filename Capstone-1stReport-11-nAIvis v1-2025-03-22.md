아래는 기존 마크다운 내용을 유지하면서 **더 자세히 기술한 2차 보고서 버전**이야. 각 항목별로 기술적/실험적 세부사항을 보강했으며, 수식은 인라인으로 처리해서 표 안에서도 안정적으로 보이도록 구성했어.


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
| **요구사항 정의** | - 기본 DF 모델 구현 및 학습<br>- Task 간 학습 정보 유지 가능하도록 EWC 모듈 설계<br>- Closed-world (95개 클래스), Open-world (50+45 Task) 시나리오 모두 적용 가능<br>- 학습 결과에 대한 정량적 평가 도구 포함 |
| **전체 시스템 구성** | - DF 모델 학습 및 추론 모듈<br>- EWC 손실 계산 및 중요도 저장 모듈<br>- 실험 환경 제어 모듈 (Task 분할, 결과 저장 등)<br>- 시각화 및 리포트 자동 생성기<br>- 외부 라이브러리: TensorFlow, NumPy, Matplotlib 등 |
| **주요엔진 및 기능 설계** | - 1D-CNN 모델 구조: Embedding Layer → Conv1D (128 filters, kernel=8, ReLU) → Global Average Pooling → Dense (Softmax)<br>- 입력 데이터: Tor 트래픽 방향 시퀀스 (길이 10,000)<br>- EWC 모듈: 이전 Task의 파라미터 θ* 저장 및 중요도(FIM) 계산 후 손실에 규제항 추가<br>- 실험 제어: Task 분할 자동화, 클래스별 학습 분배, 결과 CSV 저장 |
| **주요 기능의 구현** | - EWC 기반 지속 학습 구현: FIM 계산 후 손실 함수에 다음 항 추가 → $L_{total} = L_{task} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$<br>- Task별 학습 자동화: Python 스크립트로 Task 데이터 분할 및 반복 학습 프로세스 구현<br>- 학습 전략: 첫 Task는 70 epoch, 이후 Task는 20 epoch으로 점진적 학습 적용<br>- 시나리오 지원: Closed-world / Open-world 모두 대응<br>- 성능 평가 지표: Task별 Accuracy, Average Accuracy, Forgetting Rate(F), Final Accuracy<br>- 결과 저장 및 시각화: 정확도 및 망각 지표를 시각화하고 CSV로 자동 저장 |
| **기타** | - 학습 및 실험 환경: GPU 기반의 TensorFlow 환경에서 수행<br>- 각 실험은 reproducibility 확보를 위해 random seed 고정<br>- 실험 결과는 성능 비교표 및 그래프 형태로 보고서에 포함됨 |
