# 지속 학습을 적용한 치명적 망각 해결과 Deep Fingerprinting model 강화
## 1. Fingerprinting 적용 - 데이터셋 및 코드 설명

### 1-a. BigEnough 데이터셋 소개
- **데이터셋 개요**:
  - 모니터링 집합: 95개의 웹사이트에서 총 19,000개의 샘플(95 * 200)을 포함.
    - 각 사이트의 10개 서브페이지를 20회씩 방문하여 200개의 샘플 생성.
    - Open PageRank Initiative에서 가장 인기 있는 사이트를 기준으로 선정.
  - 비모니터링 집합: 상위 웹사이트의 무관한 인덱스 페이지 중 무작위로 샘플링된 19,000개의 샘플로 구성.
- **데이터셋 구성**:
  - X: 패킷 시퀀스의 방향 데이터(수신/발신).
  - y: 접속한 사이트 레이블.

- **코드 설명**:
  - utility.ipynb: 파일에서 데이터를 로드하고 NumPy 배열로 변환하는 기능을 수행
  - Model.ipynb: 1차원 CNN 아키텍처(DFNet)를 정의하며, 입력 데이터의 형태와 클래스 수를 기반으로 웹 트래픽 분류와 같은 작업에 사용할 수 있는 케라스 모델을 생성
  - NonBase_ClosedWorld.ipynb: EWC를 아예 적용하지 않고 두 개의 태스크를 차례대로 학습 시킨 후 성능 평가 -> EWC_ClosedWorld.ipynb와의 결과와 비교하여 CL의 성능 입증
  - EWC.ipynb: EWC loss를 계산하기 위한 함수와 train 함수를 정의
  - EWC_ClosedWorld.ipybn: EWC를 적용하여 두 개의 태스크를 차례대로 학습 시킨 후 성능 평가
  - ClosedWorld.ipynb: 두 개의 태스크를 하나의 태스크로 합쳐 한 번에 학습 시킨 후 성능 평가 -> joint 성능 측정
---

### 1-b. BigEnough 데이터셋으로 EWC 효과 입증
- **실험 설정**:
  - Task A: 0~42 레이블 분류.
  - Task B: 43~94 레이블 분류.
- **실험 절차**:
  1. Task A 데이터셋으로 모델 학습.
  2. 같은 모델을 Task B 데이터셋으로 학습(Task A의 중요한 파라미터가 덮어씌워짐).
  3. Task A와 Task B 데이터셋으로 테스트하여 성능 측정.
- **실험 결과**:
  - EWC 미적용(Non-based):
    - Task B 학습 후 Task A 정확도: **0.0**.
  - EWC 적용:
    - Task A의 train dataset으로 모델을 학습시키는 과정에서 EWC loss값을 계산하고, Task B의 train dataset으로 모델을 학습시킬 때 EWC loss를 반영
    - Task B 학습 시 EWC 손실 항목 추가: **0.388** 정확도.
  - **결론**: Website Fingerprinting 도메인에서 Continual Learning의 효과 입증.

---

### 1-c. 성능 개선
- EWC의 효과는 입증되었으나 절대적인 성능은 낮음.
- **파라미터 튜닝**:
  - Epoch: Task A와 Task B 학습 반복 횟수 조정.
  - Alpha: Task A의 중요한 파라미터를 유지하기 위한 규제 강도 조정.
  - Sequence length: 각 패킷 시퀀스의 길이 설정.

---

## 2. 성능 개선 방안

### 2-1. 목적
- **목표**: 사용가능한(실용적인) 모델 제안:
  - 학계에 제안 가능한 수준의 모델 및 방법론.
  - 실제 네트워크 환경(OpenWorld)에서 적용 가능.

---

### 2-2. Loss Function 수정
- **추가 내용**:
  - L1 norm 등 추가적인 손실 항목 추가.
  - 손실 함수의 전체적인 정규화(balancing).

---

### 2-3. 코드의 구조적 수정
- 현재 MNIST에 적용된 EWC 코드를 기반으로 작성된 모델.
- **개선 방향**:
  - 다른 연구에서 사용된 EWC 모델 분석 및 적용 (예: Malware 탐지, Anomaly Server 탐지).
  - 이를 Fingerprinting 도메인에 맞게 통합.

---

## 3. Contribution 설정

### 교수님 피드백 (중간 발표):
- 새로운 기술이나 연구 방법론 포함:
  1. 손실 함수 수정: 커스텀 loss term 추가.
  2. 데이터 전처리 방법 수정: Normalization 및 Regularization 적용.
  3. 시기 차이가 존재하는 데이터셋에 맞는 동적 가중치 스케일링 적용.
  4. EWC + SI 조합 실험:
     - SI buffer 데이터를 추가해 이전 작업 데이터 학습 강화.
  5. 기존 EWC 접근법과 비교해 Fingerprinting 도메인에서 Catastrophic Forgetting을 얼마나 줄였는지 공헌 지점을 명확히 함.

---

## 4. 목표 학회
KICS 한국통신학회 학회지(하계종합학술발표회 - 학부논문)
- 논문 제출 마감일: 5월 중순(작년은 5/17)
- A4 1-2페이지
- 졸업 프로젝트팀들 포스터 발표 위주
- https://conf.kics.or.kr/
- https://conf.kics.or.kr/journalRegister#undergraduate

