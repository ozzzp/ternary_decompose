# Ternary Singular Value Decomposition as a Better Parameterized Form in Linear Mapping

Code of papers [Ternary Singular Value Decomposition as a Better Parameterized Form in Linear Mapping](https://arxiv.org/abs/2308.07641).

A simple yet novel parameterized form of linear mapping to achieves remarkable network compression performance: a pseudo SVD called Ternary SVD (TSVD). 
Unlike vanilla SVD, TSVD limits the $U$ and $V$ matrices in SVD to ternary matrices form in $\{\pm 1, 0\}$. This means that instead of using the expensive multiplication instructions, TSVD only requires addition instructions when computing $U(\cdot)$ and $V(\cdot)$.

## Current Result

| Model | Method | $\times$(B) | $+$(B) | params (M) | tern params (M) | top-1(\%)|
| --- | --- | --- | ---  | --- | --- | --- |
| ConvNeXt-T | original | 4.47 | 4.46 | 28.6 | 0 | 82.07 |
|   | $1\%$ tol TSVD | 0.074 | 12.6 | 0.23 | 279.3 | 82.05|
|   | $7\%$ tol TSVD (F) | 0.046 | 6.12  | 0.15 | 129.78 | 82.04|
| Swin-T | original | 4.50 | 4.50 | 28.3 | 0 | 80.40 |
|   | $1\%$ tol TSVD | 0.21 | 12.3 | 0.29 | 261.8 | 80.37|
|   | $7\%$ tol TSVD (F) | 0.18 | 5.84 | 0.19 | 109.03 | 80.34|
| ResNet-50 | original | 4.10 | 4.10 | 25.6 | 0 | 75.85 |
|    | $1\%$ tol TSVD | 0.060 | 10.83 | 0.21 | 242.5 | 75.81|
|   | $7\%$ tol TSVD (F) | 0.035 | 4.99 | 0.16 | 105.23 | 75.79|

BERT and GLUE:

| Method | $\times$/+(B) |  P / TP (M)| CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE |
| --- | --- | --- | ---  | --- | --- | --- |--- |--- |--- |--- |
| original | 11.19 / 11.18  | 109 / 0 |  59.33 | 92.78 | 89.19 / 84.55 | 87.52 / 87.23 | 87.50 / 90.81 | 83.79 / 84.27 | 90.61 |64.26 |
| $1\%$ tol TSVD | 0.34 / 29.40 | 23 / 825 | 60.81 | 92.43 | 89.03 / 83.57 | 88.47 / 88.28 | 87.42 / 90.68 | 83.50 / 84.36 | 90.57 | 65.70 |
| $5\%$ tol TSVD |  0.33 / 15.88 | 23 / 440  | 60.65 | 91.05 | 89.78 / 85.04 | 87.57 / 87.40 | 86.71 / 89.51 | 83.11 / 82.75 | 89.36 |61.01 |

OPT-6.7B:
| Method | $\times$(T) | +(T) | params (B) | tern params (B) | wikitext2 | ptb | c4 |
|---|---|---|---|---|---|---|---|
| original | 14.72 | 14.72 | 6.86 | 0 |  10.86 |  13.08 | 11.74 |
| $1\%$ tol TSVD | 1.11 | 31.98 | 0.22 | 55.03 | 11.10 | 13.73 | 12.16 |
| $1.5\%$ tol TSVD | 1.11 | 27.66 | 0.22 | 47.37 | 12.12 | 15.62 | 13.34| 
| $2\%$ tol TSVD | 1.11 | 24.64 | 0.22 | 42.00 | 19.08 | 26.06 |  25.75 |

## How to Use

Package dependency:
```
pytorch >= 1.10
```

### Use as a Python Package
First, make sure `tern_svd` and `torchprofile` path is included in the `PYTHONPATH` environment. 
```python
import torch
from tern_svd import *
from torchprofile import count_mul_and_add_for_first_input

with replace_Linear_to_ternary_SVD_linear():
    model = ...         # create a pytorch model and loading parameter here.

# define a ternary transition function.
trans_fun = transform_policy(
    steps=20, 
    tolerance=1e-2, 
    verbose=True, 
    cos_thresh=0.8386)     

@tern_svd_layer_patch
def trans(M):
    M.weight_to_usv(trans_fun, None, prune_rate=float('Inf'))
    del M.weight # After transition, if there is no weight in ternary SVD layer, it will be forward as USV(.)

model.apply(trans)
print(model)

# count mul and add instructions for the first time of model running
model = count_mul_and_add_for_first_input(model)

# eval model
...
```

### QAT training
QAT usage is quit similar with PTQ, but using `transform_policy_for_QAT` instead of `transform_policy` and do ternary transition for every time after upgrading parameters.
```python
import torch
from tern_svd import *
from torchprofile import count_mul_and_add_for_first_input

with replace_Linear_to_ternary_SVD_linear():
    model = ...         # create a pytorch model and loading parameter here.

# using `transform_policy_for_QAT` instead of `transform_policy`
trans_fun, s_fun = transform_policy_for_QAT(
    steps=20, 
    tolerance=1e-2, 
    verbose=False
    cos_thresh=0.8386)

@tern_svd_layer_patch
def trans(M):
    M.weight_to_usv(trans_fun, 
                    s_fun, 
                    prune_rate=5)
    # keep M.weight and M.rest_weight for QAT training.

...

for step, batch in enumerate(train_dataloader):
    ... # Do training here
    optimizer.step()
    model.apply(trans) # Do ternary transition
    
```

### Multi-GPU Usage via torch.dist

The Ternary SVD alogrithm limit that the weight matrix should be deployed on one GPU. Hence, we only support data parallel and pipeline parallel paradigm. For LLM, we recommed to use pipeline parallel, while for small model like Swin-T or BERT, data parallel is still OK but need additional sync and broadcast communications. 

There are some auxiliary function in `tern_svd/dist.py` to do such additional communications. For their usage, see our experiment file in `experiment/`, e.g.  `run_imagenet.py`.

### Repeat Experiment
For repeat experiment in `experiment/`, additional package is required to get model and dataset:
```
transformer >= 4.29.1
timm >= 0.9.2
datasets >= 2.12
accelerate >= 0.19.0
```
then
```bash
$ cd experiment/
$ python run_imagenet_script.py
$ python run_opt_script.py
$ python run_glue_script.py
```

## Dictionary Structure

```bash
$ ls ./*
 ./readme.md

./experiment:                   # main python scripts of experiment in papers 
conv_profile.py
datautils.py
playground.py
profile.py
proof.py
run_glue.py
run_glue_prune_ofa.py
run_glue_prune_ofa_script.py
run_glue_script.py              # main script of GLUE and BERT experiment
run_imagenet.py
run_imagenet_script.py          # main script of ImageNet1k experiment.
run_opt.py
run_opt_script.py               # main script of OPT experiment
tsvd.py

./tern_svd:                     # main package of ternary decompose.
__init__.py                     # top API file, including context manager that replace all linear layers by ternary SVD layer
base.py                         # base layer module of ternary svd layers
dist.py                         # torch.dist communications utilities.
tern_svd_conv2d.py              # ternary svd convolution layers
tern_svd_conv2d_transpose.py    # ternary svd transpose convolution layers
tern_svd_linear.py              # ternary svd linear layers
ternary_decompose_jax.py        # main algorithm implementations by jax, but its version iteration is much slower than pytorch version.
ternary_decompose_pytorch.py    # main algorithm implementations by pytorch.

./torchprofile:                 # a modified torchprofile package that can seperately stat the mul and add instructions as Ternary SVD required.
__init__.py
handlers.py
profile.py
drwxrutils
version.py
```

## Copyright

This project was referenced and modified some code from existing project, we here list and express our respects to these authors:

| path | link |
| ---- | ---- |
| `./torchprofile` | [torchprofile](https://github.com/zhijian-liu/torchprofile)|
| `./experiment/datautils.py` | [GPTQ](https://github.com/IST-DASLab/gptq)|
| `./experiment/run_opt.py` | [GPTQ](https://github.com/IST-DASLab/gptq)|
|`./experiment/run_imagenet.py`| [HuggingFace Example](https://github.com/huggingface/transformers/tree/main/examples/pytorch)|
|`./experiment/run_glue.py`| [HuggingFace Example](https://github.com/huggingface/transformers/tree/main/examples/pytorch)|

## Citation
```
@misc{chen2023ternary,
      title={Ternary Singular Value Decomposition as a Better Parameterized Form in Linear Mapping}, 
      author={Boyu Chen and Hanxuan Chen and Jiao He and Fengyu Sun and Shangling Jui},
      year={2023},
      eprint={2308.07641},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Todo

- [ ] custom CUDA kernel for 2-bit storage
- [ ] 2:4 sparsity instruction remould
