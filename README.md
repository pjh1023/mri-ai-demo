# 의료영상에서 AI는 무엇을 볼까? — 고교 특강 시연 노트북

진짜 뇌 MRI에서 **종양을 찾는 AI**와 **노이즈에서 MRI를 그려내는 AI**를 코드 한 칸씩 실행하며 만들어 보는 시연 자료입니다.
(POSTECH 인공지능대학원 김원화 교수 연구실, 지역 이공계 기초역량 강화 프로그램 고교 특강용)

## 바로 열어 보기 (설치 없이, 브라우저에서)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pjh1023/mri-ai-demo/blob/main/MRI_AI_demo_colab.ipynb)

위 버튼을 누르고 `런타임 → 모두 실행`을 누르면 끝입니다. Google 계정만 있으면 됩니다. (첫 셀이 데이터와 모델을 자동으로 받아옵니다.)

## 내 컴퓨터에서 실행하기

```bash
pip install -r requirements.txt
jupyter lab MRI_AI_demo.ipynb
```
GPU는 필요 없습니다. 전체 실행에 약 2분 30초 걸립니다 (GPU가 있으면 MRI 그리기 파트가 자동으로 GPU를 씁니다).

## 흐름

1. 진짜 뇌 MRI 3,064장 열어 보기 — 컴퓨터에게 사진은 숫자표
2. 사람이 규칙 정하기("밝으면 종양" + 3개 더) vs AI가 예시에서 배우기
3. 종양 찾기 U-Net — 학습 전 → 현장에서 짧게 학습 → 1·5·20·300바퀴 단계별 결과 → 직접 해보기 → 틀린 사례
4. diffusion으로 노이즈에서 새 MRI 그리기 → 만든 뇌를 종양 찾기 AI에 넣어 보기
5. 결과 그림을 ChatGPT·Gemini에 보여주고 물어보기 (브라우저에서)

## 파일

| 파일 | 용도 |
|---|---|
| `MRI_AI_demo.ipynb` | 시연용 노트북 (발표자 노트북에서 실행) |
| `MRI_AI_demo_colab.ipynb` | 같은 내용의 Google Colab 버전 (백업·학생 배포용) |
| `MRI_AI_demo_executed.ipynb` | 실행 완료본 (결과 미리 보기) |
| `진행가이드.pdf` | 20분 진행 타임라인, 멘트, 예상 질문, 챗봇 답변 예시 |
| `diffusion_model.py` | diffusion 모델 정의 |
| `data/` | 공개 뇌종양 MRI 3,064장 (128×128) + 종양 마스크 |
| `models/` | 미리 학습한 파라미터 (종양 찾기 최종 + 1·5·20바퀴 중간 단계, diffusion) |

## 데이터 출처

Cheng, J. et al. (2015). *brain tumor dataset*. figshare. https://doi.org/10.6084/m9.figshare.1512427 (CC BY 4.0)
T1 조영증강 뇌 MRI 3,064장 (환자 233명, 수막종·신경교종·뇌하수체종양), 종양 마스크 포함.
