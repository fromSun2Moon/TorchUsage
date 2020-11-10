# config.py
데이터 학습에 필요한 변수들을 저장하고 사용하는 파일.

# model.py 
데이터를 학습시키기 위한 모델들이 정의되어 있는 파일.

# train.py 
data를 읽어서 모델을 구성하고 학습시키는 파일 (train, eval).

# utils.py
학습에 필요한 커스텀 데이터셋 정의 및 스케줄러가 저장된 파일.

# test.py
pth file들로 모델 test를 진행해서 csv 를 쓰는 파일.

# data
학습할 데이터가 저장된 디렉토리.

# bin
train한 weights 파일들이 존재하는 디렉토리.
0.pth ~ 9.pth : pth files.

--------------------------------------------------------
> 아래 파일들은 Dacon 홈페이지에서 다운 가능합니다.

# test.csv
test csv file

# sample_submission
result file을 쓰기 위한 frame file.

# MAE.ipynb
test 정답 데이터가 없는 경우 모델 overfitting 점검.