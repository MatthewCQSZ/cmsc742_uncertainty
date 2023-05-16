from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DotmapUtils import get_required_argument
from config.utils import swish, get_affine_params, TrainingState

import gym
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from enn.losses import base as losses_base
from enn import networks
import optax
import haiku as hk
import jax
import jax.numpy as jnp

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#get gpu device for jax allocation
assert len(jax.devices("gpu")) > 0
gpu = jax.devices("gpu")[0]


class PtModel(nn.Module):

    def __init__(self, ensemble_size, in_features, out_features):
        super().__init__()

        self.num_nets = ensemble_size

        self.in_features = in_features
        self.out_features = out_features

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, in_features, 500)

        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, 500, 500)

        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, 500, 500)

        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, 500, out_features)

        self.inputs_mu = nn.Parameter(torch.zeros(in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(in_features), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)

    def compute_decays(self):

        lin0_decays = 0.0001 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.00025 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.00025 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.0005 * (self.lin3_w ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    def fit_input_stats(self, data):

        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(TORCH_DEVICE).float()

    def forward(self, inputs, ret_logvar=False):

        # Transform inputs
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b

        mean = inputs[..., :self.out_features // 2]

        logvar = inputs[..., self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)


class CartpoleConfigModule:
    ENV_NAME = "MBRLCartpole-v0"
    TASK_HORIZON = 200
    NTRAIN_ITERS = 50
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 6, 4
    GP_NINDUCING_POINTS = 200

    # Create and move this tensor to GPU so that
    # we do not waste time moving it repeatedly to GPU later
    ee_sub = torch.tensor([0.0, 0.6], device=TORCH_DEVICE, dtype=torch.float)
    ee_sub_jnp = jax.device_put(jnp.array([0.0, 0.6]), gpu)

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1,
            }
        }

    @staticmethod
    def obs_preproc(obs):
        if isinstance(obs, np.ndarray):
           return np.concatenate([np.sin(obs[:, 1:2]), np.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                obs[:, 1:2].sin(),
                obs[:, 1:2].cos(),
                obs[:, :1],
                obs[:, 2:]
            ], dim=1)
        else:
            return jnp.concatenate([jnp.sin(obs[:, 1:2]), jnp.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def obs_cost_fn(obs):
        ee_pos = CartpoleConfigModule._get_ee_pos(obs)

        if isinstance(obs, torch.Tensor):
            ee_pos -= CartpoleConfigModule.ee_sub
            ee_pos = ee_pos ** 2
            ee_pos = - ee_pos.sum(dim=1)
            return - (ee_pos / (0.6 ** 2)).exp()
        else:
            ee_pos -= CartpoleConfigModule.ee_sub_jnp
            ee_pos = ee_pos ** 2
            ee_pos = -jnp.sum(ee_pos, axis=1)
            return - jnp.exp(ee_pos / (0.6 ** 2))

    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, torch.Tensor):
            return 0.01 * (acs ** 2).sum(dim=1)
        else:
            return 0.01 * jnp.sum((acs ** 2), axis=1)

    @staticmethod
    def _get_ee_pos(obs):
        x0, theta = obs[:, :1], obs[:, 1:2]

        if isinstance(x0, torch.Tensor):
            return torch.cat([
                x0 - 0.6 * theta.sin(), -0.6 * theta.cos()
            ], dim=1)
        else:
            return jnp.concatenate([x0 - 0.6 * jnp.sin(theta), -0.6 * jnp.cos(theta)], axis=1)

    def nn_constructor(self, model_init_cfg):

        ensemble_size = get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size")

        load_model = model_init_cfg.get("load_model", False)

        assert load_model is False, 'Has yet to support loading model'

        model = PtModel(ensemble_size,
                        self.MODEL_IN, self.MODEL_OUT * 2).to(TORCH_DEVICE)
        # * 2 because we output both the mean and the variance

        model.optim = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # TODO: flexible parameters
        seed = 0
        model.rng = hk.PRNGSequence(seed)
        
        model.enn = networks.MLPEnsembleMatchedPrior(
            output_sizes=[50, 50, self.MODEL_OUT * 2],
            dummy_input = np.zeros((32, self.MODEL_IN)),
            num_ensemble=ensemble_size,
        )
        
        index = model.enn.indexer(next(model.rng))
        model.enn_params, model.enn_network_state = model.enn.init(next(model.rng), np.zeros((32, 6)), index) #rng, inputs, index
        
        
        
        # Optimizer
        model.enn_optimizer = optax.adam(1e-3)
        model.opt_state = model.enn_optimizer.init(model.enn_params)
        model.enn_state = TrainingState(model.enn_params, model.enn_network_state, model.opt_state)
        
        def model_apply(x,
                        index,):
            net_out, state = model.enn.apply(model.enn_state.params, model.enn_state.network_state, x, index)
            net_out = networks.parse_net_output(net_out)
            mean = net_out[:,:4]
            logvar = net_out[:,4:]
            return mean, logvar

        model.enn_apply = model_apply
        
        class LogLoss(losses_base.SingleLossFnArray):
            def __call__(self,
                            params: hk.Params,
                            state: hk.State,
                            x,
                            y,
                            index, ):
                    net_out, state = model.enn.apply(params, state, x, index)
                    net_out = networks.parse_net_output(net_out)
                    mean = net_out[:,:4]
                    logvar = net_out[:,4:]
                    inv_var = jnp.exp(-logvar)      
                    train_losses = ((mean - y) ** 2) * inv_var + logvar
                    return jnp.mean(train_losses), net_out
                
        model.enn_loss_fn = LogLoss()
        return model


CONFIG_MODULE = CartpoleConfigModule
