# UniformAugment

Unofficial PyTorch Reimplementation of [UniformAugment](https://arxiv.org/abs/2003.14348). Most of codes are from [Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment) and [PyTorch RandAugment](https://github.com/ildoonet/pytorch-randaugment).

## Introduction
UniformAugment is an automated data augmentation approach that completely avoids a search phase.

## Usage

```python
from torchvision.transforms import transforms
from UniformAugment import UniformAugment

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
])
# Add UniformAugment with num_ops hyperparameter (num_ops=2 is optimal)
transform_train.transforms.insert(0, UniformAugment())
```

## Experiment

The details of the experiment were consulted with the authors of the UniformAugment paper.

You can run an example experiment with, 

```bash
$ python UniformAugment/train.py -c confs/wresnet28x10_cifar.yaml --dataset cifar10 \
    --save cifar10_wres28x10.pth --dataroot ~/data --tag v1
```

### CIFAR-10 Classification

| Model             | Paper's Result | Ours         |
|-------------------|---------------:|-------------:|
| Wide-ResNet 28x10 | 97.33          | 97.30        |
| Shake26 2x96d     | 98.1           | In progress  |

### CIFAR-100 Classification

| Model             | Paper's Result | Ours         |
|-------------------|---------------:|-------------:|
| Wide-ResNet 28x10 | 82.82          | In progress  |
| Wide-ResNet 40x2  | 79.01          | 79.01        |



### ImageNet Classification

| Model             | Paper's Result | Ours         |
|-------------------|---------------:|-------------:|
| ResNet-50         | 77.63          | In progress  |
| ResNet-200        | 80.4           | In progress  |


## Core class
```python
class UniformAugment:
    def __init__(self, ops_num=2):
        self._augment_list = augment_list(for_autoaug=False)
        self._ops_num = ops_num

    def __call__(self, img):
        # Selecting unique num_ops transforms for each image would help the
        #   training procedure.
        ops = random.choices(self._augment_list, k=self._ops_num)

        for op in ops:
            augment_fn, low, high = op
            probability = random.random()
            if random.random() < probability:
                img = augment_fn(img.copy(), random.uniform(low, high))

        return img
```


## References

- UniformAugment : [Paper](https://arxiv.org/abs/2003.14348)
- Fast AutoAugment : [Code](https://github.com/kakaobrain/fast-autoaugment) [Paper](https://arxiv.org/abs/1905.00397)
- Pytorch RandAugment: [Code](https://github.com/ildoonet/pytorch-randaugment)
