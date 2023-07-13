# Fake or Real: AI 생성 이미지 판별 경진대회

- https://aiconnect.kr/competition/detail/227/task/295/taskInfo
- 2023/06/26 ~ 2023/07/06
- AI 생성 이미지 판별 경진대회

### ✔ 대회 개최 배경 및 컨셉

생성 AI 기술의 발전으로 가짜 이미지를 누구나 손쉽게 만들 수 있게 되면서, 잘못된 이미지나 동영상을 만들어 정보를 퍼뜨리거나, 신원을 도용하는 등 악용하는 사례가 발생하고 있습니다. 이에 따라, 생성 AI가 만든 가짜 이미지와 진짜 이미지를 정교하게 구분하는 AI 모델의 발전이 필요합니다.

이번 대회를 통해 참가자들은 보다 정확하고 신뢰성 높은 이미지 인식 기술 개발을 촉진하고 딥러닝, 컴퓨터 비전 등의 기술적 발전에 기여할 수 있습니다. 이미지 분석 및 인식, 인공지능 모델 개발 등의 기술적인 능력을 통해 인공지능 기술의 한계와 발전 방향을 다른 커넥터와 함께 고민해보는 기회가 될 것입니다.

이번 대회에 참여하여 생성형 AI 가 만들어낸 새로운 시대에서 발생할 수 있는 사회적 문제를 고민해보고, 내가 가진 머신 러닝 기술력으로 문제를 해결해보세요!

### 평가지표

- macro f1 score

	- $F1 score = 2 * \frac{Precision * Recall}{Precision + Recall}$

---
#### 프로젝트 구조
	
	├── data/
	│   ├── train/
	│   │   ├── fake_images/
	│   │   │   └── fake_00000.png
 	│   │	│   └── fake_00001.png
	│   │   └── real_images/
	│   │   │   └── real_0000.png
  	│   │	│   └── real_00001.png
	│   ├── valid/
 	│   │   ├── fake_images/
	│   │   │   └── fake_00003.png
 	│   │	│   └── fake_00004.png
	│   │   └── real_images/
	│   │   │   └── real_00003.png
  	│   │	│   └── real_00004.png
	│   └── test/
 	│   │   ├── images/
  	│   │   │   └── test_00000.png
	│       └── sample_submission.csv
	├── runs/
 	│ 
	└── efficientB6.ipynb/

---
# 팀 차동민

|이름|닉네임|프로필|
|:--|:---|:-----|
|차형석|hsmaro||
|김동하|Eastha0526|https://github.com/Eastha0526|
|권경민|km0228kr||

---

# 사용 기법

- Albumentation을 통한 Data Augumentation
	
	```python
	train_transform = A.Compose([
	                    A.Resize(224, 224),
	                    A.AdvancedBlur(),
	                    A.ColorJitter(),
	                    A.GaussNoise(),
	                    A.OpticalDistortion(distort_limit=(-0.3, 0.3), shift_limit=0.5, p=0.5),
	                    A.HorizontalFlip(),
	                    A.Affine(scale=(0.9, 2), translate_percent=(-0.1, 0.1), rotate=(-10, 10), shear=(-20,20)),
	                    A.ElasticTransform(alpha=300),
	                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
	                    ToTensorV2()
	                    ])
	```
 - 이번 대회에서는 이미지가 생성된 이미지인지 진짜 이미지인지 예측하는 대회이기에 데이터 증강을 통하여 정확도를 높이려고 시도하였다.

- efficientnet finetunning을 통한 예측
	```python
	class Efficientnet_b6(nn.Module):

	    def __init__(self, num_classes=2):
		super(Efficientnet_b6, self).__init__()
		self.backbone = models.efficientnet_b6(pretrained=True)
		self.dropout = nn.Dropout(p=0.5)
		self.classifier = nn.Linear(1000, num_classes)
	
	    def forward(self, x):
		x = self.backbone(x)
		x = self.dropout(x)
		out = self.classifier(x)
	
		return out

	model = Efficientnet_b6()
	```

- efficientnet의 경우 타 모델(ResNet50, VCG16)등의 모델 보다 높은 정확도와 적은 파라미터, 연산 수를 가지고 있기 때문에 선택하였고 컴퓨팅 사양(VRAM)의 크기를 적절하게 판단하여 efficientnetb6를 사용하여 fine-tunning을 사용해 trasnfer learning을 사용하였다.

- Optimizer
	- Optimizer를 구성할 때, 추가적인 라이브러리인 torch_optimizer와 pytorch에서 제공해주는 optimizer를 사용하여 모델을 학습하였고 추가적으로 Warmup Scheduler를 사용하여 Global minia에 수렴할 수 있도록 구성하였다.
	- Warmup Scheduler
		- https://arxiv.org/abs/1812.01187
		- Bag of Tricks for Image Classification with Convolutional Neural Networks
			- 위의 논문에서 Warmup Scheduler에 대하여 초기에는 작은 learning rate로 튜닝을 하다 learning rate를 update를 하는 방식으로 모델 적합에 사용하였다.
	
	   	```python
		# Warmup Scheduler
		class WarmupLR(optim.lr_scheduler.LambdaLR):
	
			def __init__(
				self,
				optimizer: optim.Optimizer,
				warmup_end_steps: int,
				last_epoch: int = -1,
			    ):
	
				def wramup_fn(step: int):
				    if step < warmup_end_steps:
					return float(step) / float(max(warmup_end_steps, 1))
				    return 1.0
	
			super().__init__(optimizer, wramup_fn, last_epoch)
		```

- 데이터를 확인하였을 때 당시 train 데이터의 개수는 약 20000개 였고 test 데이터의 경우 약 60000개 정도로 훈련 데이터보다 테스트 데이터의 경우에 예측해야할 데이터량이 더 많았다.
- 대회측에서 외부데이터 사용을 허용하여서 훈련 데이터를 늘린다면 성능 향상이 있을 것이라고 판단하여서 추가 데이터를 수집하여 훈련에 포함을 하고 데이터량을 늘렸다.
- 사용데이터
	- https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- 실제로 훈련 데이터의 수를 증가 시킨 결과 f1_score의 경우 0.82 에서 최종 0.86까지 상승하는 것을 확인할 수 있었다.

# 최종 결과 

- ![image](https://github.com/Eastha0526/fake_or_real/assets/110336043/577002ac-2ec3-412a-ab68-2a23fabf02fe)

- 115팀의 참가팀 중 최종 순위 21등 (상위 18%)

# 배운점

- 
