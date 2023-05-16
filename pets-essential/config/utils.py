import numpy as np
import tensorflow as tf
import torch
from torch import nn as nn
import haiku as hk
import optax
import jax.numpy as jnp
from typing import NamedTuple


def swish(x):
    return x * torch.sigmoid(x)


def truncated_normal(size, std):
    # We use TF to implement initialization function for neural network weight because:
    # 1. Pytorch doesn't support truncated normal
    # 2. This specific type of initialization is important for rapid progress early in training in cartpole

    # Do not allow tf to use gpu memory unnecessarily
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=cfg) as sess:
        val = sess.run(tf.compat.v1.truncated_normal(shape=size, stddev=std))

        # Close the session and free resources
        sess.close()

    return torch.tensor(val, dtype=torch.float32)


def get_affine_params(ensemble_size, in_features, out_features):

    w = truncated_normal(size=(ensemble_size, in_features, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)

    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b


def log_likelihood(x, x_hat):
    """Gaussian log-likelihood."""
    return -0.5 * jnp.sum((x - x_hat)**2) - 0.5 * x.shape[-1] * jnp.log(2 * jnp.pi)

def kl_divergence(mu, logvar):
    """KL divergence for a diagonal Gaussian distribution."""
    return -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar))

class TrainingState(NamedTuple):
  params: hk.Params
  network_state: hk.State
  opt_state: optax.OptState