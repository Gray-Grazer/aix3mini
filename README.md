# aix3miniproj

### 가상환경 구축
- miniforge3 설치 후 conda 패키지를 사용하여 가상환경 구축
- https://github.com/conda-forge/miniforge?tab=readme-ov-file
  
### 이미지 유사도 측정에 사용한 모델명
- face_recognition (ageitgey)

### Install 모듈 목록 - 반드시 가상환경에 설치해주세요
- pip install pillow
- pip install numpy==1.20.3
- pip install face_recognition

### 발생 중인 문제
1. 업로드한 이미지 파일의 갯수만큼 정상적으로 폴더를 생성한 이후 생성된 폴더에 유사도 측정범위내 이미지가 복사되지 않고 업로드된 이미지가 복사되는 문제

### 그 외 개선사항 
1. fastapi와 uvicorn을 설치 후 로컬 서버 환경 구축하여 적용
   - pip install fastapi
   - pip install uvicorn
2. 휴지통 역할을 할 폴더를 따로 생성하고 사용자가 이미제 삭제 요청 시 유사도 측정범위내 이미지를 휴지통 폴더로 복사하는 작업을 추가
