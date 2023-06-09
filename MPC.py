from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.io import savemat

from DotmapUtils import get_required_argument
from optimizers import CEMOptimizer

from tqdm import trange
from functools import partial

import torch

import optax
import haiku as hk
import jax
import jax.numpy as jnp

from config.utils import TrainingState

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f"Running on TORCH_DEVICE:{TORCH_DEVICE}")


class Controller:
    def __init__(self, *args, **kwargs):
        """Creates class instance.
        """
        pass

    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """Trains this controller using lists of trajectories.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        """Resets this controller.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def act(self, obs, t, get_pred_cost=False):
        """Performs an action.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def dump_logs(self, primary_logdir, iter_logdir):
        """Dumps logs into primary log directory and per-train iteration log directory.
        """
        raise NotImplementedError("Must be implemented in subclass.")


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


class MPC(Controller):
    optimizers = {"CEM": CEMOptimizer}

    def __init__(self, params):
        """Creates class instance.

        Arguments:
            params
                .env (gym.env): Environment for which this controller will be used.
                .ac_ub (np.ndarray): (optional) An array of action upper bounds.
                    Defaults to environment action upper bounds.
                .ac_lb (np.ndarray): (optional) An array of action lower bounds.
                    Defaults to environment action lower bounds.
                .per (int): (optional) Determines how often the action sequence will be optimized.
                    Defaults to 1 (reoptimizes at every call to act()).
                .prop_cfg
                    .model_init_cfg (DotMap): A DotMap of initialization parameters for the model.
                        .model_constructor (func): A function which constructs an instance of this
                            model, given model_init_cfg.
                    .model_train_cfg (dict): (optional) A DotMap of training parameters that will be passed
                        into the model every time is is trained. Defaults to an empty dict.
                    .model_pretrained (bool): (optional) If True, assumes that the model
                        has been trained upon construction.
                    .mode (str): Propagation method. Choose between [E, DS, TSinf, TS1, MM].
                        See https://arxiv.org/abs/1805.12114 for details.
                    .npart (int): Number of particles used for DS, TSinf, TS1, and MM propagation methods.
                    .ign_var (bool): (optional) Determines whether or not variance output of the model
                        will be ignored. Defaults to False unless deterministic propagation is being used.
                    .obs_preproc (func): (optional) A function which modifies observations (in a 2D matrix)
                        before they are passed into the model. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc (func): (optional) A function which returns vectors calculated from
                        the previous observations and model predictions, which will then be passed into
                        the provided cost function on observations. Defaults to lambda obs, model_out: model_out.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc2 (func): (optional) A function which takes the vectors returned by
                        obs_postproc and (possibly) modifies it into the predicted observations for the
                        next time step. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .targ_proc (func): (optional) A function which takes current observations and next
                        observations and returns the array of targets (so that the model learns the mapping
                        obs -> targ_proc(obs, next_obs)). Defaults to lambda obs, next_obs: next_obs.
                        Note: Only needs to process NumPy arrays.
                .opt_cfg
                    .mode (str): Internal optimizer that will be used. Choose between [CEM].
                    .cfg (DotMap): A map of optimizer initializer parameters.
                    .plan_hor (int): The planning horizon that will be used in optimization.
                    .obs_cost_fn (func): A function which computes the cost of every observation
                        in a 2D matrix.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .ac_cost_fn (func): A function which computes the cost of every action
                        in a 2D matrix.
                .log_cfg
                    .save_all_models (bool): (optional) If True, saves models at every iteration.
                        Defaults to False (only most recent model is saved).
                        Warning: Can be very memory-intensive.
                    .log_traj_preds (bool): (optional) If True, saves the mean and variance of predicted
                        particle trajectories. Defaults to False.
                    .log_particles (bool) (optional) If True, saves all predicted particles trajectories.
                        Defaults to False. Note: Takes precedence over log_traj_preds.
                        Warning: Can be very memory-intensive
        """
        super().__init__(params)
        self.dO, self.dU = params.env.observation_space.shape[0], params.env.action_space.shape[0]
        self.ac_ub, self.ac_lb = params.env.action_space.high, params.env.action_space.low
        self.ac_ub = np.minimum(self.ac_ub, params.get("ac_ub", self.ac_ub))
        self.ac_lb = np.maximum(self.ac_lb, params.get("ac_lb", self.ac_lb))
        self.update_fns = params.get("update_fns", [])
        self.per = params.get("per", 1)

        self.model_init_cfg = params.prop_cfg.get("model_init_cfg", {})
        self.model_train_cfg = params.prop_cfg.get("model_train_cfg", {})
        self.prop_mode = get_required_argument(params.prop_cfg, "mode", "Must provide propagation method.")
        self.npart = get_required_argument(params.prop_cfg, "npart", "Must provide number of particles.")
        self.ign_var = params.prop_cfg.get("ign_var", False) or self.prop_mode == "E"

        self.obs_preproc = params.prop_cfg.get("obs_preproc", lambda obs: obs)
        self.obs_postproc = params.prop_cfg.get("obs_postproc", lambda obs, model_out: model_out)
        self.obs_postproc2 = params.prop_cfg.get("obs_postproc2", lambda next_obs: next_obs)
        self.targ_proc = params.prop_cfg.get("targ_proc", lambda obs, next_obs: next_obs)

        self.opt_mode = get_required_argument(params.opt_cfg, "mode", "Must provide optimization method.")
        self.plan_hor = get_required_argument(params.opt_cfg, "plan_hor", "Must provide planning horizon.")
        self.obs_cost_fn = get_required_argument(params.opt_cfg, "obs_cost_fn", "Must provide cost on observations.")
        self.ac_cost_fn = get_required_argument(params.opt_cfg, "ac_cost_fn", "Must provide cost on actions.")

        self.save_all_models = params.log_cfg.get("save_all_models", False)
        self.log_traj_preds = params.log_cfg.get("log_traj_preds", False)
        self.log_particles = params.log_cfg.get("log_particles", False)

        #coefficient for adding a bonus for exploring states with high epistemic uncertainty
        self.epistemic_coef = params.opt_cfg.get("epi_coef", 0.0)
        self.epistemic_aux = self.epistemic_coef > 0
        print("Epistemic Aux", self.epistemic_aux)

        #bool for whether epinet
        self.epinet = params.opt_cfg.get("epinet", True)
        print("Epinet", self.epinet)

        # Perform argument checks
        assert self.opt_mode == 'CEM'
        assert self.prop_mode == 'TSinf', 'only TSinf propagation mode is supported'
        assert self.npart % self.model_init_cfg.num_nets == 0, "Number of particles must be a multiple of the ensemble size."

        # Create action sequence optimizer
        opt_cfg = params.opt_cfg.get("cfg", {})
        self.eval_optimizer = CEMOptimizer(
            sol_dim=self.plan_hor * self.dU,
            lower_bound=np.tile(self.ac_lb, [self.plan_hor]),
            upper_bound=np.tile(self.ac_ub, [self.plan_hor]),
            cost_function=self._compile_cost_eval,
            **opt_cfg
        )

        self.train_optimizer = CEMOptimizer(
            sol_dim=self.plan_hor * self.dU,
            lower_bound=np.tile(self.ac_lb, [self.plan_hor]),
            upper_bound=np.tile(self.ac_ub, [self.plan_hor]),
            cost_function=self._compile_cost_train,
            **opt_cfg
        )

        # Controller state variables
        self.has_been_trained = params.prop_cfg.get("model_pretrained", False)
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])
        self.train_in = np.array([]).reshape(0, self.dU + self.obs_preproc(np.zeros([1, self.dO])).shape[-1])
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1]
        )

        print("Created an MPC controller, prop mode %s, %d particles. " % (self.prop_mode, self.npart) +
              ("Ignoring variance." if self.ign_var else ""))

        if self.save_all_models:
            print("Controller will save all models. (Note: This may be memory-intensive.")
        if self.log_particles:
            print("Controller is logging particle predictions (Note: This may be memory-intensive).")
            self.pred_particles = []
        elif self.log_traj_preds:
            print("Controller is logging trajectory prediction statistics (mean+var).")
            self.pred_means, self.pred_vars = [], []
        else:
            print("Trajectory prediction logging is disabled.")

        #make sure jax has gpu
        assert len(jax.devices("gpu")) > 0
        self.gpu = jax.devices("gpu")[0]

        # Set up pytorch model
        self.model = get_required_argument(
            params.prop_cfg.model_init_cfg, "model_constructor", "Must provide a model constructor."
        )(params.prop_cfg.model_init_cfg)
        
        

    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.

        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            acs_trajs: A list of action matrices, actions in rows.
            rews_trajs: A list of reward arrays.

        Returns: None.
        """

        # Construct new training points and add to training set
        new_train_in, new_train_targs = [], []
        for obs, acs in zip(obs_trajs, acs_trajs):
            new_train_in.append(np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1))
            new_train_targs.append(self.targ_proc(obs[:-1], obs[1:]))
        self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
        self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

        # Train the model
        self.has_been_trained = True


        # Train the pytorch model
        self.model.fit_input_stats(self.train_in)

        idxs = np.random.randint(self.train_in.shape[0], size=[self.model.num_nets, self.train_in.shape[0]])

        epochs = self.model_train_cfg['epochs']

        # TODO: double-check the batch_size for all env is the same
        batch_size = 32

        epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

        for _ in epoch_range:

            for batch_num in range(num_batch):
                batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]

                loss = 0.01 * (self.model.max_logvar.sum() - self.model.min_logvar.sum())
                loss += self.model.compute_decays()

                if self.epinet:
                    train_in = jax.device_put(self.train_in[batch_idxs], self.gpu)[0]
                    train_targ = jax.device_put(self.train_targs[batch_idxs], self.gpu)[0]
                    self.index = self.model.enn.indexer(next(self.model.rng))

                    grads, net_out = jax.grad(self.model.enn_loss_fn, has_aux=True)(self.model.enn_state.params, 
                                                            self.model.enn_state.network_state, 
                                                            train_in,
                                                            train_targ, 
                                                            self.index)
                    updates, new_opt_state = self.model.enn_optimizer.update(grads, self.model.enn_state.opt_state)
                    new_params = optax.apply_updates(self.model.enn_state.params, updates)
                    new_state = TrainingState(
                            params=new_params,
                            network_state=self.model.enn_state.network_state,
                            opt_state=new_opt_state,
                        )
                    
                    self.model.enn_state = new_state

                    # TODO: add ENN output to base network output
                else:
                    train_in = torch.from_numpy(self.train_in[batch_idxs]).to(TORCH_DEVICE).float()
                    train_targ = torch.from_numpy(self.train_targs[batch_idxs]).to(TORCH_DEVICE).float()
                    mean, logvar = self.model(train_in, ret_logvar=True)

                    inv_var = torch.exp(-logvar)
                    
                    train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
                    train_losses = train_losses.mean(-1).mean(-1).sum()
                    # Only taking mean over the last 2 dimensions
                    # The first dimension corresponds to each model in the ensemble

                    loss += train_losses

                    self.model.optim.zero_grad()
                    loss.backward()
                    self.model.optim.step()
                
            idxs = shuffle_rows(idxs)

            #val_in = torch.from_numpy(self.train_in[idxs[:5000]]).to(TORCH_DEVICE).float()
            #val_targ = torch.from_numpy(self.train_targs[idxs[:5000]]).to(TORCH_DEVICE).float()

            #mean, _ = self.model(val_in)
            #mse_losses = ((mean - val_targ) ** 2).mean(-1).mean(-1)

            #epoch_range.set_postfix({
            #    "Training loss(es)": mse_losses.detach().cpu().numpy()
            #})

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.eval_optimizer.reset()
        self.train_optimizer.reset()

        for update_fn in self.update_fns:
            update_fn()

    def act(self, obs, t, train = False, get_pred_cost=False):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """
        if not self.has_been_trained:
            return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action

        self.sy_cur_obs = obs

        if train:
            soln = self.train_optimizer.obtain_solution(self.prev_sol, self.init_var)
        else:
            soln = self.eval_optimizer.obtain_solution(self.prev_sol, self.init_var)
        self.prev_sol = np.concatenate([np.copy(soln)[self.per * self.dU:], np.zeros(self.per * self.dU)])
        self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)

        return self.act(obs, t)

    def dump_logs(self, primary_logdir, iter_logdir):
        """Saves logs to either a primary log directory or another iteration-specific directory.
        See __init__ documentation to see what is being logged.

        Arguments:
            primary_logdir (str): A directory path. This controller assumes that this directory
                does not change every iteration.
            iter_logdir (str): A directory path. This controller assumes that this directory
                changes every time dump_logs is called.

        Returns: None
        """
        # TODO: implement saving model for pytorch
        # self.model.save(iter_logdir if self.save_all_models else primary_logdir)
        if self.log_particles:
            savemat(os.path.join(iter_logdir, "predictions.mat"), {"predictions": self.pred_particles})
            self.pred_particles = []
        elif self.log_traj_preds:
            savemat(
                os.path.join(iter_logdir, "predictions.mat"),
                {"means": self.pred_means, "vars": self.pred_vars}
            )
            self.pred_means, self.pred_vars = [], []
    
    @torch.no_grad()
    def get_epistemic_info_rad(self, inputs):
        #we will use Information Radius at first ex. https://mathoverflow.net/questions/244293/generalisations-of-the-kullback-leibler-divergence-for-more-than-two-distributio
        #need to figure out how to efficiency compute info radius over many different particles
        #consider during eval, to compute optimal path with mean of all transition functions, not a single one
        if self.epinet:
            #inputs = inputs[None, :]
            #inputs_tiled = jnp.tile(inputs[:, None, :, :], (1, self.model.num_nets, 1, 1))
            mean, var = self.model.enn_apply(inputs, self.index)
            average_mean = jnp.mean(mean)
            average_var = jnp.mean(var)


            diffs_sq = jnp.square(mean - average_mean)
            k = mean.shape[-1]
            log_term = jnp.sum(jnp.log(jnp.sqrt(average_var)) - jnp.log(jnp.sqrt(var)))

            #from https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
            kl_d = 0.5 * (log_term - k + jnp.sum(((var + diffs_sq) / average_var)))

            info_rad = jnp.mean(kl_d)

            
        else:
            inputs_tiled = torch.tile(inputs[:, None, :, :], (1, self.model.num_nets, 1, 1))
            mean, var = self.model(inputs_tiled)
            average_mean = torch.mean(mean, dim=1)
            average_var = torch.mean(var, dim=1)


            diffs_sq = torch.square(mean - average_mean)
            k = mean.shape[-1]
            log_term = torch.sum(torch.log(torch.sqrt(average_var)) - torch.log(torch.sqrt(var)), dim=-1)

            #from https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
            kl_d = 0.5 * (log_term - k + torch.sum(((var + diffs_sq) / average_var), dim=-1))

            info_rad = torch.mean(kl_d, dim=1).flatten()

        return info_rad
    

    @torch.no_grad()
    def _compile_cost_torch(self, ac_seqs, rng1, rng2, epi_rew = False):

        nopt = ac_seqs.shape[0]

        ac_seqs = torch.from_numpy(ac_seqs).float().to(TORCH_DEVICE)

        # Reshape ac_seqs so that it's amenable to parallel compute
        # Before, ac seqs has dimension (400, 25) which are pop size and sol dim coming from CEM
        ac_seqs = ac_seqs.view(-1, self.plan_hor, self.dU)
        #  After, ac seqs has dimension (400, 25, 1)

        transposed = ac_seqs.transpose(0, 1)
        # Then, (25, 400, 1)

        expanded = transposed[:, :, None]
        # Then, (25, 400, 1, 1)

        tiled = expanded.expand(-1, -1, self.npart, -1)
        # Then, (25, 400, 20, 1)

        ac_seqs = tiled.contiguous().view(self.plan_hor, -1, self.dU)
        # Then, (25, 8000, 1)
        cur_obs = torch.from_numpy(self.sy_cur_obs).float().to(TORCH_DEVICE)
        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(nopt * self.npart, -1)

        costs = torch.zeros(nopt, self.npart, device=TORCH_DEVICE)

        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]
            next_obs, inputs = self._predict_next_obs_vanilla(cur_obs, cur_acs)

            cost = self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs)
            

            if self.epistemic_aux:
                cost -= self.epistemic_coef * self.get_epistemic_info_rad(inputs)

                cost = cost.view(-1, self.npart)
                
            cost = torch.reshape(cost, costs.shape)

            costs += cost
            cur_obs = self.obs_postproc2(next_obs)

        # Replace nan with high cost
        costs[costs != costs] = 1e6
        retVal = costs.mean(dim=1).detach().cpu()

        return retVal
    
    @partial(jax.jit, static_argnums=(0, 4))
    def _compile_cost_jax(self, ac_seqs, rng1, rng2, epi_rew = False):

        nopt = ac_seqs.shape[0]

        # Expand current observation

        ac_seqs = jax.device_put(jnp.array(ac_seqs), self.gpu)
        ac_seqs = jax.lax.stop_gradient(ac_seqs) 
        ac_seqs = jnp.reshape(ac_seqs, (-1, self.plan_hor, self.dU))
        transposed = jnp.transpose(ac_seqs, (1, 0, 2))
        expanded = transposed[:, :, None]
        tiled = jnp.repeat(expanded, self.npart, 2)
        ac_seqs = jnp.reshape(tiled, (self.plan_hor, -1, self.dU))

        cur_obs = jnp.array(self.sy_cur_obs)
        cur_obs = jax.lax.stop_gradient(cur_obs) 
        cur_obs = jax.device_put(cur_obs, self.gpu)
        cur_obs = cur_obs[None]
        cur_obs = jnp.repeat(cur_obs, nopt * self.npart, axis=0)

        costs = jax.device_put(jnp.zeros((nopt, self.npart)), self.gpu)
        costs = jax.lax.stop_gradient(costs)


        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]
            next_obs, inputs = self._predict_next_obs_epinet(cur_obs, cur_acs, rng1, rng2)

            cost = self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs)
            

            if self.epistemic_aux:
                cost -= self.epistemic_coef * self.get_epistemic_info_rad(inputs)


            cost = cost.reshape(-1, self.npart)

            costs += cost
            cur_obs = self.obs_postproc2(next_obs)

        # Replace nan with high cost
        costs = jnp.nan_to_num(costs, copy=False, nan=1e6)
        retVal = jnp.mean(costs, axis=1)

        return retVal

    @torch.no_grad()
    def _compile_cost_eval(self, ac_seqs):
        if self.epinet:
            return np.array(self._compile_cost_jax(ac_seqs, next(self.model.rng), next(self.model.rng), epi_rew=False))
        else:
            return np.array(self._compile_cost_torch(ac_seqs, next(self.model.rng), next(self.model.rng), epi_rew=False))

    @torch.no_grad()
    def _compile_cost_train(self, ac_seqs):
        if self.epinet:
            return np.array(self._compile_cost_jax(ac_seqs, next(self.model.rng), next(self.model.rng), epi_rew=False))
        else:
            return np.array(self._compile_cost_torch(ac_seqs, next(self.model.rng), next(self.model.rng), epi_rew=False))

    def _predict_next_obs_vanilla(self, obs, acs):
        proc_obs = self.obs_preproc(obs)

        assert self.prop_mode == 'TSinf'

        proc_obs = self._expand_to_ts_format(proc_obs)
        acs = self._expand_to_ts_format(acs)

        inputs = torch.cat((proc_obs, acs), dim=-1)

        mean, var = self.model(inputs)

        predictions = mean + torch.randn_like(mean, device=TORCH_DEVICE) * var.sqrt()

        # TS Optimization: Remove additional dimension
        predictions = self._flatten_to_matrix(predictions)

        return self.obs_postproc(obs, predictions), inputs

    def _predict_next_obs_epinet(self, obs, acs, rng1, rng2):
        proc_obs = self.obs_preproc(obs)

        assert self.prop_mode == 'TSinf'

        proc_obs = self._expand_to_ts_format(proc_obs)
        acs = self._expand_to_ts_format(acs)

        inputs = jnp.concatenate((proc_obs, acs), axis=-1)[0]

        index = self.model.enn.indexer(rng1)

        enn_out, network_state = self.model.enn.apply(self.model.enn_state.params, 
                                      self.model.enn_state.network_state, 
                                      inputs, 
                                      index) # params, state, inputs, index


        enn_out = enn_out.train + enn_out.prior
        model_out = enn_out.shape[-1]
        mean = enn_out[..., :model_out//2]
        var = enn_out[..., model_out//2:]
        predictions = mean + jax.random.normal(rng2, var.shape) * jnp.sqrt(var)

        # TS Optimization: Remove additional dimension
        predictions = self._flatten_to_matrix(predictions)

        return self.obs_postproc(obs, predictions), inputs

    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]

        # Before, [10, 5] in case of proc_obs
        if self.epinet:
            reshaped = mat.reshape(-1, self.model.num_nets, self.npart // self.model.num_nets, dim)
            transposed = jnp.transpose(reshaped, axes=(1, 0, 2, 3))
            reshaped = mat.reshape(self.model.num_nets, -1, dim)
        else:
            reshaped = mat.view(-1, self.model.num_nets, self.npart // self.model.num_nets, dim)
            # After, [2, 5, 1, 5]
            transposed = reshaped.transpose(0, 1)
            # After, [5, 2, 1, 5]
            reshaped = transposed.contiguous().view(self.model.num_nets, -1, dim)
            # After. [5, 2, 5]

        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):

        if self.epinet:
            dim = ts_fmt_arr.shape[-1]
            reshaped = ts_fmt_arr.reshape(self.model.num_nets, -1, self.npart // self.model.num_nets, dim)
            transposed = jnp.transpose(reshaped, (1, 0, 2, 3))
            reshaped = transposed.reshape(-1, dim)

        else:
            dim = ts_fmt_arr.shape[-1]

            reshaped = ts_fmt_arr.view(self.model.num_nets, -1, self.npart // self.model.num_nets, dim)

            transposed = reshaped.transpose(0, 1)

            reshaped = transposed.contiguous().view(-1, dim)

        return reshaped
