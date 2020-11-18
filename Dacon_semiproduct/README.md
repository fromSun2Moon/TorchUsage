# Dacon Semiproduct code


```
C:.
│  .gitignore
│  config.py
│  README.md
│  test.py
│  train.py
│
├─bin
├─data
│      test.csv
│      train.csv
│      train1.csv
│      val.csv
│
├─ipynb
│      MAE.ipynb
│      split_file.ipynb
│
├─src
│      get_score.py
│      model.py
│      utils.py
│
└─__pycache__
        config.cpython-37.pyc
        model.cpython-37.pyc
        utils.cpython-37.pyc
```

1. 먼저 데이콘에서 데이터를 다운로드 받아, data 디렉토리에 넣어줍니다.
- https://dacon.io/competitions/official/235554/data/
- train.csv, test.csv
- sample_submission : 결과를 쓰기 위한 csv file.

2. split_file.ipynb를 키고 한 행마다 실행하여 학습 데이터를 얻습니다.
3. config.py에서 데이터 경로와 원하는 하이퍼 파라미터를 설정합니다.
4. train.py에서 모델을 학습시킵니다.
5. 학습 후, test.py로 모델을 테스합니다.

## config.py
데이터 학습에 필요한 변수들을 저장하고 사용하는 파일.

## model.py 
데이터를 학습시키기 위한 모델들이 정의되어 있는 파일.

## train.py 
data를 읽어서 모델을 구성하고 학습시키는 파일 (train, eval).

## utils.py
학습에 필요한 커스텀 데이터셋 정의 및 스케줄러가 저장된 파일.

## test.py
pth file들로 모델 test를 진행해서 csv 를 쓰는 파일.

## data
학습할 데이터가 저장된 디렉토리.

## bin
train한 weights 파일(.pth/.bin/.pt)들이 존재하는 디렉토리.

--------------------------------------------------------

## MAE.ipynb
test 정답 데이터가 없는 경우 모델 overfitting 점검.

