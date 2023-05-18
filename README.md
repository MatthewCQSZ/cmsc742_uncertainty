# Code Base for *Towards Better Uncertainty Estimation in Model-Based Reinforcement Learning*

Based on PyTorch implementation of [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://github.com/quanvuong/handful-of-trials-pytorch) and JAX implementation of [Epistemic Neural Networks](https://github.com/deepmind/enn).

The code is tested on Python3.11 and GPU only, note that JAX requires CUDA 11.1 or above. 

Running experiments requires mujoco, a helpful guide on how to set up mujoco can be found [here](https://github.com/MatthewCQSZ/transfer-drl/tree/main/install_instructions).

To install dependencies, run
```
pip install -r requirements.txt
pip install git+https://github.com/deepmind/enn
```

To replicate experiments, simply run:
```
python mbexp.py -env cartpole
```

Other supported environments include `halfcheetah` `pusher` `reacher` and `mountain_car`.

Use `-epinet` flag to train on Epinet, otherwise it trains on regular PETS as default.

Use `-epi_coef` to toggle the epistemic rewards.