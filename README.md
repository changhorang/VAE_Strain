# VAE_Strain (time series forcasting)

## 1-1. Model 변경 (Transfomer Encoder)

    - 기존의 LSTM을 활용해 time series forcasting을 Transformer Encoder으로 예측
    
    - 기대 효과는 Transformer에 있는 self-attention의 장점을 이용해 예측의 향상

## 1-2. VAE 적용 (Transfomer_Encoder_VAE)

    - 극단값을 더 잘 잡아낼 수 있는 효과를 기대

    - Optimizer는 Adam을 이용

## 2. 결과
    
    - 전체적인 경향성은 각 parameter별로 따라가는 것으로 확인은 되었지만, 예측된 값은 아직 큰 차이가 있는 것으로 확인

    - 과거 값을 이용하는 경우, n_past는 30정도가 제일 양호하게 결과가 나옴 (batch_szie=200, epoch=200 기준)

    - 극단값은 조금씩 잡는듯 하지만, 아직은 모델이 추세를 따라가는 수준으로 성능이 좋지 못한 것으로 보임

    - VAE loss(MSELoss+KLloss)를 반영하였지만, 예측값이 오히려 더 안나오는 결과가 나옴 (KL loss가 잘 잡히지 않았거나, Transformer Encoder에는 VAE 적용이 조금 어렵다고 생각됨)

    - MSE Loss만 적용 시에는 극단값을 조금 더 잘 잡아내는 수준

![figure_epoch201_past30_batch200.png](./figure_save/figure_epoch201_past30_batch200.png)

![VAE_figure_epoch100_past30_batch200.png](./figure_save/figure_epoch100_past30_batch200.png)
    
## 3. More..

    - VAE를 통해서 나온 샘플들이 극단값을 잘 잡아내는지 확인

    - VAE loss(MSELoss+KLloss)를 반영하였지만, 예측값이 오히려 더 안나오는 결과가 나옴 (KL loss가 잘 잡히지 않았거나, Transformer Encoder에는 VAE 적용이 조금 어렵다고 생각됨)

    - MSE Loss만 적용 시에는 극단값을 조금 더 잘 잡아내는 수준




