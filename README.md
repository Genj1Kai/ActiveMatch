# ActiveMatch

Code for the paper: ["ActiveMatch: End-to-end Semi-supervised Active Representation Learning"](https://arxiv.org/abs/2110.02521) by Xinkai Yuan, Zilinghan Li, and Gaoang Wang.

## Installation

```
conda env create --name activematch --file environment.yml
conda activate activematch
python train.py
```


## Running


### Example

Train our model with 40 to 200 labels of CIFAR-10:

```python
python train.py --dataset cifar10 --num-labeled 40 --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@40-200 --stop-active 200 --num-sample 32 --epoch-warmup 15
```

Train our model with 200 to 1000 labels of CIFAR-100:

```python
python train.py --dataset cifar100 --num-labeled 200 --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@200-1000 --stop-active 1000 --num-sample 4 --epoch-warmup 15
```

## Citing this work

```latex
@misc{yuan2021activematch,
      title={ActiveMatch: End-to-end Semi-supervised Active Representation Learning}, 
      author={Xinkai Yuan and Zilinghan Li and Gaoang Wang},
      year={2021},
      eprint={2110.02521},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
