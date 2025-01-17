# Image distored assessement score (DLS&P2P)

## Overview
* 연구 주제: 인물 이미지 내의 왜곡 탐지 및 정도 평가
* 목표: 이미지 보정으로 인한 왜곡을 탐지하고 수치화하는 새로운 평가 지표 제안
* 주요 특징: 직선 유사도(DLS)와 포인트 추적(P2P) 기반의 평가 방식 제안

## Background and Motivation
* 사회적 배경
  * 사진 기술 및 보정 기술의 발달로 인한 이미지 왜곡 증가
  * 신분증명 문서에서의 왜곡으로 인한 문제 발생
  * 과도한 보정으로 인한 외모 일반화 및 사회적 부작용
* 기술적 배경
  * 기존 보정 알고리즘 평가를 위한 지표 부족
  * 벤치마크 데이터셋 부재
  * 왜곡 평가 법안을 뒷받침할 기술적 근거 필요

## Method

### DISTORTED LINE SIMILARITY (DLS)
![Image](https://github.com/user-attachments/assets/913051f3-ea35-4474-bb2f-d27a9108f214)

* 동작 과정:
  1. Edge Detection을 통한 직선 정보 추출
  2. 원본-보정 이미지 간 직선 정보 차이 계산
  3. 임계값 이상 차이나는 픽셀 추출
  4. 전체 대비 왜곡 픽셀 비율 계산

### POINT TO POINT (P2P)
![Image](https://github.com/user-attachments/assets/a0172144-ccaa-44c6-901d-1756dd0995dd)


* 동작 과정:
  1. Motion Tracking으로 코너 검출
  2. 원본-보정 이미지 간 코너 움직임 계산
  3. 움직임의 평균값으로 왜곡 정도 수치화

## Directory Structure
```
📦 Image-Distortion-Assessment
├── 📄 IDEA.py            # IDEA 및 P2P 구현
├── 📄 PSNR_SSIM.py       # PSNR, SSIM 비교 구현
└── 📂 data/              # 테스트 이미지
```

## Usage

### IDEA & P2P Score 계산
```python
from IDEA import IDEA, P2P
from PIL import Image
import cv2

# IDEA Score 계산
original = Image.open('path/to/original.jpg')
distorted = Image.open('path/to/distorted.jpg')
idea_score = IDEA(original, distorted)

# P2P Score 계산
original = cv2.imread('path/to/original.jpg')
distorted = cv2.imread('path/to/distorted.jpg')
p2p_score = P2P(original, distorted)
```

### PSNR & SSIM Score 계산
```python
from PSNR_SSIM import PSNR, SSIM

psnr_score = PSNR(original, distorted)
ssim_score = SSIM(original, distorted)
```

## Experimental Results

### 평가 지표 성능 비교
* DLS와 P2P는 기존 SSIM, PSNR 대비 80% 더 효과적한 왜곡 감지
* 신체 부위별, 왜곡 강도별 평가에서 우수한 성능 입증
* MD 이미지를 통한 평가 지표의 신뢰성 검증

### COCO-based Dataset
![Image](https://github.com/user-attachments/assets/c14d06a5-642b-4593-9ca2-54bcf7cc1f01)

* COCO 2017 데이터셋 기반
* 다양한 신체 부위 (얼굴, 어깨, 엉덩이, 다리, 전신)
* 다양한 왜곡 강도 (+50, +100, -50, -100)
* MD Image를 통한 비교 실험 지원

## Contributions
* 새로운 이미지 왜곡 평가 지표 (DLS, P2P) 제안
* 실제 사용 사례에 기반한 벤치마크 데이터셋 구축
* 기존 평가 지표 대비 향상된 왜곡 감지 성능
* 이미지 리터치 기술 개발 및 평가를 위한 프레임워크 제공