from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from time import localtime, strftime

from dotmap import DotMap
from scipy.io import savemat
from tqdm import trange

from Agent import Agent
from DotmapUtils import get_required_argument

from torch.utils.tensorboard import SummaryWriter


class MBExperiment:
    def __init__(self, params):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.

                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
        """

        # Assert True arguments that we currently do not support
        assert params.sim_cfg.get("stochastic", False) == False

        self.env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        self.agent = Agent(DotMap(env=self.env, noisy_actions=False))

        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )
        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        self.ninit_rollouts = params.exp_cfg.get("ninit_rollouts", 1)
        self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")

        self.logdir = os.path.join(
            get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory."),
            strftime("%Y-%m-%d--%H:%M:%S", localtime()) + str(params.sim_cfg.env) + "_epi_coef_" + str(self.policy.epistemic_coef),
            
        )
        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 1)
        self.ntrain = params.log_cfg.get("ntrain", 1)
        self.writer = SummaryWriter(self.logdir)

    def run_experiment(self):
        """Perform experiment.
        """
        os.makedirs(self.logdir, exist_ok=True)

        traj_obs, traj_acs, traj_rets, traj_rews = [], [], [], []

        # Perform initial rollouts
        samples = []
        for i in range(self.ninit_rollouts):
            samples.append(
                self.agent.sample(
                    self.task_hor, self.policy
                )
            )
            traj_obs.append(samples[-1]["obs"])
            traj_acs.append(samples[-1]["ac"])
            traj_rews.append(samples[-1]["rewards"])

        if self.ninit_rollouts > 0:
            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples]
            )

        # Training loop
        for i in trange(self.ntrain_iters):
            print("####################################################################")
            print("Starting training iteration %d." % (i + 1))

            iter_dir = os.path.join(self.logdir, "train_iter%d" % (i + 1))
            os.makedirs(iter_dir, exist_ok=True)

            train_samples = []
            eval_samples = []

            for j in range(max(self.ntrain, self.nrollouts_per_iter)):
                train_samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy, train=True
                    )
                )

            for j in range(max(self.neval, self.nrollouts_per_iter)):
                eval_samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy, train=False
                    )
                )
            print("Rewards obtained eval:", [sample["reward_sum"] for sample in eval_samples[:self.neval]])
            print("Rewards obtained train:", [sample["reward_sum"] for sample in eval_samples[:self.neval]])
            #traj_obs.extend([sample["obs"] for sample in samples[:self.nrollouts_per_iter]])
            #traj_acs.extend([sample["ac"] for sample in samples[:self.nrollouts_per_iter]])
            #traj_rets.extend([sample["reward_sum"] for sample in samples[:self.neval]])
            #traj_rews.extend([sample["rewards"] for sample in samples[:self.nrollouts_per_iter]])
            train_samples = train_samples[:self.nrollouts_per_iter]
            eval_samples = eval_samples[:self.nrollouts_per_iter]

            #old logging, we changed to tensorboard
            #self.policy.dump_logs(self.logdir, iter_dir)
            #savemat(
            #    os.path.join(self.logdir, "logs.mat"),
                #{
            #        "observations": traj_obs,
            #        "actions": traj_acs,
            #        "returns": traj_rets,
            #        "rewards": traj_rews
            #    }
            #)

            #per eval iter logging
            sum_return = 0
            for j in range(len(train_samples)):
                sum_return += train_samples[j]["reward_sum"]
            mean_return = sum_return / len(train_samples)
            self.writer.add_scalar("mean train return vs train iter", mean_return, i)

            sum_return = 0
            for j in range(len(eval_samples)):
                sum_return += eval_samples[j]["reward_sum"]
            mean_return = sum_return / len(eval_samples)
            self.writer.add_scalar("mean eval return vs train iter", mean_return, i)

            # Delete iteration directory if not used
            if len(os.listdir(iter_dir)) == 0:
                os.rmdir(iter_dir)

            if i < self.ntrain_iters - 1:
                self.policy.train(
                    [sample["obs"] for sample in train_samples],
                    [sample["ac"] for sample in train_samples],
                    [sample["rewards"] for sample in train_samples]
                )
