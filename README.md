# VAE_Strain (time series forcasting)

## 1-1. Model 변경 (Transfomer Encoder)

    - 기존의 LSTM을 활용해 time series forcasting을 Transformer Encoder으로 예측
    
    - 기대 효과는 Transformer에 있는 self-attention의 장점을 이용해 예측의 향상

## 1-2. VAE 적용 (Transfomer_Encoder_VAE)

    - 극단값을 더 잘 잡아낼 수 있는 효과를 기대

    - Optimizer는 Adam을 이용

## 2. 결과
    
    - 전체적인 경향성은 각 parameter별로 따라가는 것으로 확인은 되었지만, 예측된 값은 아직 큰 차이가 있는 것으로 확인

    - 사용한 Data file은 01000002.txt이며, 진행된 결과는 figure_save에 저장된 이미지와 같다 (추가 ~~예정~~).

    - 과거 값을 이용하는 경우, n_past는 30정도가 제일 양호하게 결과가 나옴 (batch_szie=200, epoch=200 기준)

    - 극단값은 어느정도 잡는듯 하지만, 추세를 따라가는 수준 및 성능이 좋지 못한 것으로 보임

![figure_epoch201_past30_batch200.png](./figure_save/figure_epoch201_past30_batch200.png)
    
## 3. More..

    ~~- 아직 train을 진행하는 과정에 있으며, 전체 파일에 대한 결과 확인이 필요 (data import 시키는 코드 작업 진행중...)~~

    ~~- argparse를 이용한 parameter들이 수정이 가능하도록 코드 수정 예정 (2021.11.01 기준)~~ (반영 완료)

    - VAE를 통해서 나온 샘플들이 극단값을 잘 잡아내는지 확인

    - VAE에 대한 학습이 명확하게 하여 모델 리뷰 필요 (loss 반영)


