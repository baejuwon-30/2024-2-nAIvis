# Team-Info
| 항목 | 내용 |
|:---  |---  |
| 과제명 | Elastic Weight Consolidation 기반 Continual Learning을 활용한 Deep Fingerprinting 성능 개선 |
| 팀 번호 / 팀 이름 | 11-nAIvis |
| 팀 구성원 | 배주원 (리더), 신유진, 이서연 |
| 팀 지도교수 | 오세은 교수님 |
| 과제 분류 | 연구 과제 |
| 과제 키워드 | Catastrophic Forgetting, Continual Learning, Deep Fingerprinting |
| 과제 내용 요약 | Tor 네트워크 환경에서 Deep Fingerprinting(DF) 모델의 성능 저하 문제를 Continual Learning(CL)의 Elastic Weight Consolidation(EWC)을 통해 해결하여 실사용 가능한 DF 모델 개발 및 평가 |

# Project-Summary
| 항목 | 내용 |
|:---  |---  |
| 문제 정의 | 기존 DF 모델은 정적 데이터 학습 기반으로 동적 트래픽 변화 대응 한계 → Catastrophic Forgetting 발생 |
| 기존연구와의 비교 | 기존 연구 대비 CL 기반 EWC 적용, 실시간 대응, 지속 학습 가능성, 재학습 비용 절감 |
| 제안 내용 | 1D-CNN 기반 DF 모델에 EWC를 적용, Fisher Information으로 중요 가중치를 보호하는 규제항 추가, Closed/Open-world 환경 실험 |
| 기대효과 및 의의 | 보안 분야 CL 기법 확장, 실시간 WF 모델 가능성 입증, 비용 절감, 학술 기여 |
| 주요 기능 리스트 | Tor 트래픽 수집 및 전처리, 1D-CNN 기반 DF 모델 구현, EWC 통합 및 중요도 계산, 실험 자동화 스크립트, 성능 분석 및 시각화 |

# Project-Design & Implementation
| 항목 | 내용 |
|:---  |---  |
| 요구사항 정의 | 기본 DF 모델 학습, EWC 기반 지속 학습 모듈 적용, Closed/Open-world 시나리오, 정확도 유지 |
| 전체 시스템 구성 | DF 모델 모듈, EWC 적용 모듈, 실험 제어 모듈, 외부 도구(Tensorflow 등) 활용 |
| 주요엔진 및 기능 설계 | 1D-CNN 기반 DF 모델 설계, Fisher Information 기반 EWC 모듈 설계, 성능 평가 구조 설계<br>- 1D-CNN 구조 설계 및 구현<br>- 입력 데이터(패킷 방향 시퀀스, 길이 10,000)를 Embedding Layer로 처리<br>- Convolution Layer: 필터 수 128, 커널 크기 8, 활성화 함수 ReLU 적용<br>- Global Average Pooling을 사용하여 차원 축소 후 Dense Layer(소프트맥스)를 통해 최종 웹사이트 분류 |
| 주요 기능의 구현 | **EWC 모델 구현**: Fisher Information Matrix를 계산하여 손실 함수에 적용하여 중요 파라미터 보호<br>- 기존 Task 학습 후 Fisher Information Matrix(FIM) 계산<br>- 각 파라미터의 중요도를 평가하여 높은 중요도를 가지는 파라미터의 변경 최소화<br>- 손실 함수(Loss Function)에 규제항으로 반영:  
  $$ L_{total} = L_{task} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2 $$<br>- Python 스크립트를 활용한 Task별 데이터 분할 및 자동화된 학습 프로세스 구현<br>- 첫 번째 Task는 epoch 70, 이후 Task는 epoch 20으로 설정하여 점진적 학습 구현<br>- 실험 환경에서 Closed-world 및 Open-world 시나리오 동시 적용 가능<br>- 평가 지표: Task별 정확도(Accuracy), 평균 정확도(Average Accuracy), Catastrophic Forgetting 지표(F), 최종 정확도(Final Accuracy)<br>- Matplotlib를 이용하여 정확도 및 망각 현상 관련 그래프 자동 생성<br>- 결과 데이터는 CSV 형태로 로그로 저장하여 지속적 성능 추적 가능 |
| 기타 | GPU 환경에서 Python 및 Tensorflow 사용, 성능 지표 분석 |

