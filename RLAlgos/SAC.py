"""
The Soft Actor-Critic (SAC) algorithm.

! only for the ANC self-defined gym-like environment
* support continuous action space

main reference:
- CleanRL doc: https://docs.cleanrl.dev/rl-algorithms/sac/
- CleanRL codes (continuous actions): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
"""

import os
import random
import time
import datetime

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from stable_baselines3.common.buffers import ReplayBuffer

from utils.dataset_utils import MyNoiseDataset


class SAC:
    def __init__(self, env, actor_class, q_net_class, exp_name="", seed=1, load_actor_agent=None, test_dataset=None,
                 test_labels_file=None, test_snr=10, hard_weights_threshold=0.5, cuda=0, buffer_size=1000000,
                 gamma=0.99, tau=0.005, batch_size=256, policy_lr=3e-4, q_lr=1e-3, alpha_lr=1e-3, policy_frequency=2,
                 target_network_frequency=2, noise_clip=0.5, alpha=0.2, autotune=True, save_frequency=10000,
                 save_folder="./runs/"):

        self.exp_name = exp_name
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env = env

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        self.actor = actor_class(self.env).to(self.device)
        self.qf1 = q_net_class(self.env).to(self.device)
        self.qf2 = q_net_class(self.env).to(self.device)

        # * to load pre-trained models
        if load_actor_agent is not None:
            self.actor.load_state_dict(torch.load(load_actor_agent, map_location=self.device))

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr)

        # Automatic entropy tuning
        self.autotune = autotune
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha

        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            handle_timeout_termination=False,
        )

        # * SAC hyper-parameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.noise_clip = noise_clip

        # * for the tensorboard writer
        run_name = "SAC-{}-{}-{}".format(exp_name, seed,
                                         datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(exp_name, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(save_folder, run_name))

        # * for saving the best model
        self.best_episodic_return = float("-inf")
        self.save_frequency = save_frequency
        

    def learn(self, total_timesteps=1000000, learning_starts=5000):

        obs, _ = self.env.reset()
        for global_step in range(1, total_timesteps + 1):

            if global_step < learning_starts:
                # actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                actions = self.env.action_space.sample()
            else:
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).unsqueeze(0).to(self.device))
                actions = actions.detach().cpu().numpy()

                # # ! turn the soft actions to hard actions by comparing with the threshold
                # actions = (actions >= self.threshold).astype(np.int8)

            # next_obs, rewards, dones, infos = envs.step(actions)
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
            dones = terminated

            self.replay_buffer.add(obs, next_obs, actions, rewards, dones, infos)

            # + the `truncated` only controls whether to test or not
            if truncated:
                episodic_return = infos['episode_return']
                # + to save the best model (only when we use the return to test)
                if episodic_return >= self.best_episodic_return:
                    self.save(indicator="best-return")
                    self.best_episodic_return = episodic_return

                # print(f"global_step={global_step}, episodic_return={episodic_return}")
                self.writer.add_scalar("charts/episodic_return", episodic_return, global_step)

                # + save some middle models:
                if global_step % self.save_frequency == 0:
                    self.save(indicator="{}k".format(int(global_step / 1000)))

            # + manually reset the env
            if dones:
                obs, _ = self.env.reset()

            # + no need to copy, as each step will terminate
            # obs = next_obs

            # ALGO LOGIC: training.
            if global_step > learning_starts:
                data = self.replay_buffer.sample(self.batch_size)

                next_q_value = data.rewards.flatten()

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if global_step % self.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(self.policy_frequency):
                        # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self.actor.get_action(data.observations)
                        # # ! turn the pi to hard actions by comparing with the threshold
                        # pi = (pi >= self.threshold).int()
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        if self.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(data.observations)
                            alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()

                if global_step % 100 == 0:
                    self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    self.writer.add_scalar("losses/alpha", self.alpha, global_step)
                    if self.autotune:
                        self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        self.writer.close()

    def save(self, indicator="best"):
        torch.save(self.actor.state_dict(),
                   "./{}/agent-{}-{}-{}.pth".format(self.exp_name, self.exp_name, indicator, self.seed))