GPU 환경 구축 with TensorFlow

캡스톤a 11조 nAIvis

2276217 이서연

우리는 DF baseline 모델을 먼저 돌려봤다.

1. joint setting (Big Enough 내 모든 데이터셋을 한번에 전부 학습시킴)
2. non-base setting (Big Enough 내 데이터셋을 앞 절반(Task A)와 나머지 절반(Task B)로 나눠 Task A를 학습시킨 후, Task B를 연달아 학습시키고 나서 Task A, Task B classification의 accuracy를 측정)
(=Class-IL) 
3. 오픈 소스 중 mnist로 EWC 기법의 CL을 한 코드(일명 mnist EWC)를 참고한 Class-IL DF 실험

이렇게 돌려봤는데 아무래도 Big Enough 데이터셋의 방대한 크기 때문에 CPU보다는 GPU로 코드를 돌려야 시간 낭비를 줄일 수 있다는 결론에 도달했다.

우리 팀원 3명은 모두 NVIDIA GeForce GPU가 탑재된 HP OMEN 노트북을 이화여대 소프트웨어학부 테크니션에서 대여했다.

처음엔 하나의 GPU 노트북으로 코드를 돌렸는데, 실험을 진행할수록 파라미터를 바꿔가며 성능을 높여야 하는 필요성이 커져, 셋이 동시에 코드를 돌리기로 결정했다.

따라서 우리는 python, numpy, pandas, tensorflow… 등 실행 환경이 동일한  코드를 공유하기 위해 11월 21일 목요일 캡스톤a 시간에 3시간 30분 간 환경 설정을 통일하고 GPU 사용 환경을 구축했다!

아래는 GPU 환경을 구축하는 과정이다.

```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/2018e2b2-6390-45cf-9177-f8bcd34a9fce/image.png)

(참고로 우리 팀은 실험 중에는 .ipynb를 사용하고, 나중에 논문 제출 직전에 .py로 정리하여 최종으로 실험 전체를 simulate한 후 git에 업로드할 계획이다.)

먼저 위의 코드를 통해 device type을 확인해 봤다.

당연히 현재는 CPU를 사용하게 되어 있다는 걸 확인할 수 있다. 이제 GPU로 변경해보자.

## NVIDIA 드라이버 설치

GPU로 변경하려면 그래픽 카드 사양에 맞는 NVIDIA Driver를 먼저 설치해야 한다.

오멘에는 GeForce RTX 3070 Ti Laptop GPU가 탑재돼 있으므로 여기에 맞춰 세팅한 후 다운로드 한다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/149b01cb-776a-405f-9589-f1ead87f65ea/image.png)

설치 후 아나콘다 프롬프트 창에서 nvidia-smi를 입력한다.

![드라이버가 잘 설치됐다.](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/4ee80b13-862b-45e0-ad12-8e6e2e2b62fc/image.png)

드라이버가 잘 설치됐다.

## TensorFlow GPU 설치하기

아나콘다 프롬프트에서 가상환경을 하나 만들어준다.

우리 팀은 ‘nAIvis’이므로 아래와 같이 만들겠다.

```python
conda create -n nAIvis python=3.7.7
```

```python
conda activate nAIvis
pip install tensorflow-gpu==2.10.0
```

python을 3.7.7, tensorflow-gpu를 2.10.0으로 설정한 이유는 우리가 참고하는 DF 모델을 만든 논문에서 사용한 환경과 동일하게 맞추기 위함이다.

![아나콘다 프롬프트 창에서 tensorflow-gpu가 설치됐음을 확인.](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/0f654ddc-592f-4188-a0a2-74983364679a/image.png)

아나콘다 프롬프트 창에서 tensorflow-gpu가 설치됐음을 확인.

이 상태에서 다시 아래의 코드를 실행해도 아직 GPU가 추가되지 않았음을 알 수 있다.

```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

아직 cuda와 cuDNN을 설치하지 않았기 때문이다.

## CUDA와 cuDNN 설치하기

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/aed17240-37be-49f1-b61d-6b6fc502e872/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/05545231-3fd4-4286-91c3-9bd6a7832f10/image.png)

가상환경 nAIvis에 설치되어 있는 파이썬 버전(3.7.7)과 tensorflow-gpu 버전(2.10.0)에 맞춰서~

CUDA는 11.2, cuDNN은 8.1로 설치해 줄 거다.

https://www.tensorflow.org/install/source_windows?hl=ko

(→ 요 링크를 통해 어떤 버전을 다운받으면 되는지 확인해볼 수 있다.)

먼저 CUDA를 다운 받은 후

https://developer.nvidia.com/cuda-toolkit

cuDNN을 다운 받는다.

https://developer.nvidia.com/cudnn

local에 다운로드된 cuDNN을 열고 설치를 이어서 한다. 이제 NVIDIA에 회원 가입을 하면 GPU 사용 준비는 끝난다!

GPU 사용 후기: CPU로는 n시간 걸리던 코드 실행 시간이 n분으로 단축됨… GPU 짱! 비싼 장비 최고!
