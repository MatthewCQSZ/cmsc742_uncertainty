Using Python3.11, assuming you have mujoco set up
```
pip install -r requirements.txt
pip install git+https://github.com/deepmind/enn
python mbexp.py -env cartpole
python mbexp.py -env halfcheetah
python mbexp.py -env pusher
python mbexp.py -env reacher
```

Usage of ENN refers to
https://github.com/deepmind/enn/blob/01e8b6f0b003d35d02998c1d9c155c31f306e2b1/enn/supervised/sgd_experiment.py
https://colab.research.google.com/github/deepmind/enn/blob/master/enn/colabs/enn_demo.ipynb


Try this if having protobuf bug
```
pip install protobuf==3.20.*
```
