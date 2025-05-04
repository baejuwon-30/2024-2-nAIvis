아래는 요청하신 마크다운 문서 전체를 **가독성 높고 정돈된 표 형태**로 재구성한 버전입니다.  
`<br>`을 적절히 활용하고, 줄바꿈과 항목 구분을 명확히 하여 깔끔하게 보이도록 구성했습니다.

---

# Team-Info

| 항목 | 내용 |
|:--|:--|
| **과제명** | Elastic Weight Consolidation 기반 Continual Learning을 활용한 Deep Fingerprinting 성능 개선 |
| **팀 번호 / 팀 이름** | 11-nAIvis |
| **팀 구성원** | 배주원 (리더), 신유진, 이서연 |
| **팀 지도교수** | 오세은 교수님 |
| **과제 분류** | 연구 과제 |
| **과제 키워드** | Catastrophic Forgetting, Continual Learning, Deep Fingerprinting |
| **과제 내용 요약** | Tor 네트워크 환경에서 Deep Fingerprinting(DF) 모델의 성능 저하 문제를 Continual Learning(CL)의 Elastic Weight Consolidation(EWC)을 통해 해결하여 실사용 가능한 DF 모델 개발 및 평가 |

---

# Project-Summary

| 항목 | 내용 |
|:--|:--|
| **문제 정의** | 기존 DF 모델은 정적 데이터 기반으로 학습되어 동적 트래픽 변화에 대응이 어려움 → Catastrophic Forgetting 발생 |
| **기존 연구와의 비교** | 기존 연구 대비 CL 기반 EWC 적용, 실시간 대응 가능, 지속 학습 구조, 재학습 비용 절감 |
| **제안 내용** | 1D-CNN 기반 DF 모델에 EWC를 적용하여, Fisher Information을 통해 중요 가중치 보호<br>Closed/Open-world 환경에서 성능 비교 실험 수행 |
| **기대효과 및 의의** | 보안 분야에 CL 기법 적용 확장<br>실시간 Web Fingerprinting 모델 개발 가능성 입증<br>재학습 비용 절감 및 학술적 기여 |
| **주요 기능 리스트** | - Tor 트래픽 수집 및 전처리<br>- 1D-CNN 기반 DF 모델 구현<br>- EWC 통합 및 중요도 계산<br>- 실험 자동화 스크립트<br>- 성능 분석 및 시각화 도구 개발 |

---

# Project-Design & Implementation

| 항목 | 내용 |
|:--|:--|
| **요구사항 정의** | - 기본 DF 모델 학습<br>- EWC 기반 지속 학습 모듈 통합<br>- Closed/Open-world 시나리오 대응<br>- 학습 후 정확도 유지 |
| **전체 시스템 구성** | DF 모델 모듈, EWC 모듈, 실험 제어 모듈, 외부 도구 (TensorFlow 등) 활용 |
| **주요엔진 및 기능 설계** | - 1D-CNN 기반 DF 모델 구성<br>- 입력: 패킷 방향 시퀀스 (길이 10,000)<br>- Embedding Layer → Conv1D (128 filters, kernel size 8, ReLU)<br>- Global Average Pooling → Dense(Softmax)로 분류 수행<br>- EWC 모듈: Fisher Information 기반 중요도 평가 및 파라미터 보존<br>- 성능 평가를 위한 실험 구조 설계 |
| **주요 기능의 구현** |- EWC 모델 구현: 기존 Task 학습 후 Fisher Information Matrix(FIM) 계산, 중요 파라미터의 변화 억제를 위해 손실 함수에 규제항 추가
    → 최종 손실 함수: $L_{total} = L_{task} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$
- Python 스크립트를 활용한 Task별 데이터 분할 및 자동화된 학습 프로세스 구현
- 첫 Task는 epoch 70, 이후 Task는 epoch 20으로 설정하여 점진적 학습 구현
- 실험 환경에서 Closed-world 및 Open-world 시나리오 동시 적용 가능
- 평가 지표: Task별 정확도(Accuracy), 평균 정확도(Average Accuracy), Catastrophic Forgetting 지표(F), 최종 정확도(Final Accuracy)
- Matplotlib을 이용하여 정확도 및 망각 현상 관련 그래프 자동 생성
- 결과 데이터는 CSV 형태로 로그 저장하여 지속적 성능 추적 가능
 |
| **기타** | GPU 환경에서 Python 및 TensorFlow 기반 학습, 성능 지표 분석 진행 |
