import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as td
import numpy as np
import wandb
from utils import weight_init, ReplayMemory, hard_update, soft_update, FreezeParameters

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2*latent_dims)
        self.std_min = 0.1
        self.std_max = 10.0
        self.apply(weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))    
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        mean, std = torch.chunk(x, 2, -1)
        mean = 30 * torch.tanh(mean / 30)
        std = self.std_max - F.softplus(self.std_max-std)
        std = self.std_min  + F.softplus(std-self.std_min) 
        return td.independent.Independent(td.Normal(mean, std), 1)

class RecurrentModelPrior(nn.Module):
    def __init__(self, latent_dims, action_dims, hidden_dims, num_layers):
        super().__init__()
        self.recurrent_hidden_dims = hidden_dims
        self.recurrent_num_layers = num_layers
        self.std_min = 0.1
        self.std_max = 10.0

        self.state_action_embedding = nn.Sequential(
            nn.Linear(latent_dims + action_dims, hidden_dims),
            nn.ReLU()
        )

        self.rnn = nn.GRU(hidden_dims, hidden_dims, num_layers=num_layers)
        
        self.latent_decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dims, latent_dims*2)
        )

    def forward(self, z, action, h_prev, return_hidden=False):
        x = torch.cat([z, action], axis=-1)
        x = self.state_action_embedding(x)
        x, h = self.rnn(x, h_prev)
        x = self.latent_decoder(x)
        mean, std = torch.chunk(x, 2, -1)
        mean = 30 * torch.tanh(mean / 30)
        std = self.std_max - F.softplus(self.std_max-std)
        std = self.std_min  + F.softplus(std-self.std_min) 

        if return_hidden:
            return td.independent.Independent(td.Normal(mean, std), 1), h
        else:
            return td.independent.Independent(td.Normal(mean, std), 1)

    def _init_hidden_state(self, batch_size, device):
        return torch.zeros(self.recurrent_num_layers, batch_size, self.recurrent_hidden_dims).to(device)

class Critic(nn.Module):
    def __init__(self, latent_dims, action_shape):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(latent_dims + action_shape, 256),
            nn.ReLU(), nn.Linear(256, 256),
            nn.ReLU(), nn.Linear(256, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(latent_dims + action_shape, 256),
            nn.ReLU(), nn.Linear(256, 256),
            nn.ReLU(), nn.Linear(256, 1))
        self.apply(weight_init)

    def forward(self, x, a):
        x_a = torch.cat([x, a], -1)
        q1 = self.Q1(x_a)
        q2 = self.Q2(x_a)
        return q1, q2


LOG_STD_MAX = 2
LOG_STD_MIN = -5
class Actor(nn.Module):
    def __init__(self, input_shape, output_shape, env):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256) # Better result with slightly wider networks.
        self.fc2 = nn.Linear(256, 128)
        self.mean = nn.Linear(128, output_shape)
        self.logstd = nn.Linear(128, output_shape)
        # action rescaling
        self.action_scale = torch.FloatTensor(
            (env.action_space.high - env.action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (env.action_space.high + env.action_space.low) / 2.)
        self.apply(weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, eval=False):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = td.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +  1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        if eval:
            return mean
        else:
            return action, log_prob, td.independent.Independent(td.Normal(mean, std), 1)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)

class RRPCAgent(object):
    def __init__(self, env, device, num_states, num_actions, gamma, tau, env_buffer_size,
                target_update_interval, log_interval, 
                latent_dims, model_hidden_dims, kl_constraint, lambda_init, 
                alpha, alpha_autotune, seq_len, batch_size, lr):

        self.device = device 
        self.low = torch.tensor(env.action_space.low, device=device)
        self.high = torch.tensor(env.action_space.high, device=device)
        self.gamma = gamma 
        self.alpha_autotune = alpha_autotune
        self.tau = tau

        self.target_update_interval = target_update_interval
        self.log_interval = log_interval

        self.kl_constraint = kl_constraint
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.env_buffer = ReplayMemory(env_buffer_size, num_states, num_actions, np.float32)

        self.encoder = Encoder(num_states, latent_dims).to(device)
        self.model = RecurrentModelPrior(latent_dims, num_actions, model_hidden_dims, 1).to(device)
        self.actor = Actor(latent_dims, num_actions, env).to(device)
        self.critic = Critic(num_states, num_actions).to(device)
        self.critic_target = Critic(num_states, num_actions).to(device)
        hard_update(self.critic_target, self.critic)

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.log_lambda_cost = torch.tensor([np.log(lambda_init)], device=device, requires_grad=True)
        self.lambda_cost = self.log_lambda_cost.exp().item()
        self.lambda_optimizer = torch.optim.Adam([self.log_lambda_cost], lr=1e-3)

        if self.alpha_autotune:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

    def get_action(self, state, eval=False):
       
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            z = self.encoder(state).sample()     
            if eval:
                action = self.actor.get_action(z, eval)       
            else:
                action, _, _ = self.actor.get_action(z)   
        return action.cpu().numpy()[0]

    def update(self, step):
        metrics  = {}
        self.update_model(step, metrics)
        self.update_rest(step, metrics) 

        if step%self.log_interval==0:
            wandb.log(metrics, step=step)  


    def update_model(self, step, metrics):
        state_seq, action_seq, reward_seq, next_state_seq, done_seq = self.env_buffer.sample_seq(4, 64)
        state_seq = torch.FloatTensor(state_seq).to(self.device)
        next_state_seq = torch.FloatTensor(next_state_seq).to(self.device)
        action_seq = torch.FloatTensor(action_seq).to(self.device)
        reward_seq = torch.FloatTensor(reward_seq).to(self.device)
        done_seq = torch.FloatTensor(done_seq).to(self.device)  
        discount_seq = self.gamma*(1-done_seq)

        kl, z_dist, z_next_dist, _ = self.kl_loss(state_seq, action_seq, next_state_seq, step, metrics)

        self.update_dual_parameter(kl.detach(), step, metrics)
        self.encoder_opt.zero_grad()
        self.model_opt.zero_grad()
        (self.lambda_cost*kl.mean()).backward()
        self.encoder_opt.step()
        self.model_opt.step()

        self.update_dual_parameter(kl.detach(), step, metrics)

    def update_rest(self, step, metrics):
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.env_buffer.sample(self.batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)  
        discount_batch = self.gamma*(1-done_batch)

        z_dist = self.encoder(state_batch)
        with torch.no_grad():
            init_hidden_state = self.model._init_hidden_state(self.batch_size, self.device)
            z_next_prior_dist = self.model(z_dist.sample().unsqueeze(0), action_batch.unsqueeze(0), init_hidden_state)
            z_next_dist = self.encoder(next_state_batch.unsqueeze(0))
            kl = td.kl.kl_divergence(z_next_dist, z_next_prior_dist)

        critic_loss = self.critic_loss(state_batch, action_batch, reward_batch, kl.detach().squeeze(0), next_state_batch, z_next_dist.sample().squeeze(0), discount_batch, step, metrics)
        actor_loss, log_pi = self.actor_loss(state_batch, z_dist.rsample(), step, metrics)
        
        self.encoder_opt.zero_grad()
        self.critic_opt.zero_grad()
        self.actor_opt.zero_grad()
        (actor_loss + critic_loss).backward()
        self.actor_opt.step()
        self.encoder_opt.step()
        self.critic_opt.step()
        
        if step%self.target_update_interval==0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.alpha_autotune:
            self.update_alpha(log_pi, step, metrics)
        
    def kl_loss(self, state_seq, action_seq, next_state_seq, step, metrics):
        z_dist = self.encoder(state_seq)
        z_seq = z_dist.sample()

        z_next_dist = self.encoder(next_state_seq)
        init_hidden_state = self.model._init_hidden_state(64, self.device)
        z_next_prior_dist = self.model(z_seq, action_seq, init_hidden_state)

        kl = td.kl.kl_divergence(z_next_dist, z_next_prior_dist)

        if step%self.log_interval==0:
            metrics['kl_predictor'] = kl.mean().item()
            metrics['posterior_entropy'] = z_next_dist.entropy().mean().item()
            metrics['prior_entropy'] = z_next_prior_dist.entropy().mean().item()
        return kl, z_dist, z_next_dist, z_next_prior_dist
    
    def critic_loss(self, state_batch, action_batch, reward_batch, kl, next_state_batch, z_next_batch, discount_batch, step, metrics):
        with torch.no_grad():    
            next_action_batch, next_log_pi, _ = self.actor.get_action(z_next_batch)
            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_action_batch)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha*next_log_pi
            target_Q = reward_batch.unsqueeze(-1) - self.lambda_cost*kl.unsqueeze(-1) + discount_batch.unsqueeze(-1)*(target_V)
        
        Q1, Q2 = self.critic(state_batch, action_batch)
        critic_loss = (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))/2
        
        if step%self.log_interval==0:
            metrics['mean_q_target'] = torch.mean(target_Q).item()
            metrics['variance_q_target'] = torch.var(target_Q).item()
            metrics['min_q_target'] = torch.min(target_Q).item()
            metrics['max_q_target'] = torch.max(target_Q).item()
            metrics['min_true_reward'] = torch.min(reward_batch).item()
            metrics['max_true_reward'] = torch.max(reward_batch).item()
            metrics['mean_true_reward'] = torch.mean(reward_batch).item()
            metrics['critic_loss'] = critic_loss.item()
        return critic_loss

    def actor_loss(self, state_batch, z_batch, step, metrics):
        with FreezeParameters([self.critic]):
            action_batch, log_pi, action_dist = self.actor.get_action(z_batch)

            Q1, Q2 = self.critic(state_batch, action_batch)     
            Q = torch.min(Q1, Q2) - self.alpha*log_pi
        
        actor_loss = -Q.mean()

        if step%self.log_interval==0:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_entropy'] = action_dist.entropy().mean()
        return actor_loss, log_pi.detach()

    def update_dual_parameter(self, predictor_kl_, step, metrics):
        dual_loss = -self.log_lambda_cost*(torch.mean(predictor_kl_-self.kl_constraint))
        self.lambda_optimizer.zero_grad()
        dual_loss.backward()
        self.lambda_optimizer.step()
        clipped_value = torch.clone(torch.clip(self.log_lambda_cost, -np.log(1e6), np.log(1e6)))
        self.log_lambda_cost.data.copy_(clipped_value)
        self.lambda_cost = self.log_lambda_cost.exp().item()

        if step%self.log_interval==0:
            metrics['dual_loss'] = dual_loss.item()
            metrics['lambda_cost'] = self.lambda_cost

    def update_alpha(self, log_pi, step, metrics):
        alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        if step%self.log_interval==0:
            metrics['alpha_loss'] = alpha_loss.item()
            metrics['alpha'] = self.alpha