import numpy as np
import tensorflow as tf
import torch
from torch import nn as nn
import haiku as hk
import optax
import jax.numpy as jnp
from typing import NamedTuple

from enn.losses import base as losses_base
from enn import networks
import chex


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
  
  
class L2Loss(losses_base.SingleLossFnArray):
  """L2 regression applied to a single epistemic index."""

  def __call__(self,
               enn_out,
               params: hk.Params,
               state: hk.State,
               batch,
               index, ):
    """L2 regression applied to a single epistemic index."""
    net_out = networks.parse_net_output(enn_out)
    sq_loss = jnp.square(networks.parse_net_output(net_out) - batch)
    return jnp.mean(sq_loss)

class LogLoss(losses_base.SingleLossFnArray):
  """L2 regression applied to a single epistemic index."""

  def __call__(self,
               enn_out,
               params: hk.Params,
               state: hk.State,
               batch,
               index, ):
    """L2 regression applied to a single epistemic index."""
    net_out = networks.parse_net_output(enn_out)
    output = networks.parse_net_output(net_out)
    mean = output[:,:4]
    logvar = output[:,4:]
    inv_var = jnp.exp(-logvar)      
    train_losses = ((mean - batch) ** 2) * inv_var + logvar
    #train_losses = train_losses.mean(-1).mean(-1).sum()
    return jnp.mean(train_losses)
