import gym 
import random
import numpy as np
import argparse
import torch
import yaml
from pathlib import Path
import os
import wandb
import datetime
import time 
from rpc import RPCAgent 

def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='RPC')
    parser.add_argument('--env_name', type=str, default=config['env_name'])
    parser.add_argument('--device', type=str, default=config['device'])
    parser.add_argument('--seed', type=int, default=config['seed'])
    parser.add_argument('--num_train_steps', type=int, default=config['num_train_steps'])
    parser.add_argument('--learning_starts', type=int, default=config['learning_starts'])
    parser.add_argument('--gamma', type=float, default=config['gamma'])
    parser.add_argument('--tau', type=float, default=config['tau'])

    parser.add_argument('--env_buffer_size', type=int, default=config['env_buffer_size'])

    parser.add_argument('--target_update_interval', type=int, default=config['target_update_interval'])
    parser.add_argument('--log_interval', type=int, default=config['log_interval'])
    parser.add_argument('--save_snapshot_interval', type=int, default=config['save_snapshot_interval'])
    parser.add_argument('--eval_episode_interval', type=int, default=config['eval_episode_interval'])
    parser.add_argument('--num_eval_episodes', type=int, default=config['num_eval_episodes'])

    parser.add_argument('--latent_dims', type=int, default=config['latent_dims'])
    parser.add_argument('--model_hidden_dims', type=int, default=config['model_hidden_dims'])
    parser.add_argument('--model_num_layers', type=int, default=config['model_num_layers'])
    parser.add_argument('--kl_constraint', type=float, default=config['kl_constraint'])
    parser.add_argument('--lambda_init', type=float, default=config['lambda_init'])
    parser.add_argument('--alpha_autotune', type=str, default=config['alpha_autotune'])
    parser.add_argument('--alpha_init', type=float, default=config['alpha'])

    parser.add_argument('--batch_size', type=int, default=config['batch_size'])
    parser.add_argument('--lr', type=float, default=config['lr'])
    
    args = parser.parse_args()
    return args

def make_agent(env, device, args):
    num_states = np.prod(env.observation_space.shape)
    num_actions = np.prod(env.action_space.shape)
    
    if args.agent == 'RPC':
        agent = RPCAgent(env, device, num_states, num_actions, args.gamma, args.tau, 
                            args.env_buffer_size, args.target_update_interval,
                            args.log_interval, args.latent_dims, args.model_hidden_dims, 
                            args.model_num_layers, args.kl_constraint, args.lambda_init,
                            args.alpha_init, args.alpha_autotune, 
                            args.batch_size, args.lr)
    return agent

class MujocoWorkspace:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device=='cuda' else "cpu")
        self.work_dir = Path.cwd()
        self.setup()
        self.set_seeds_everywhere()
        print(self.args)
        self.agent = make_agent(self.train_env, self.device, self.args)
        self._global_step = 0
        self._global_episode = 0
        self._best_eval_returns = -np.inf

    def setup(self):
        self.train_env = gym.make(self.args.env_name)
        self.eval_env = gym.make(self.args.env_name)        
        self.robust_env = gym.make(self.args.env_name)
        
    def set_seeds_everywhere(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        self.train_env.reset(seed=self.args.seed)
        self.train_env.action_space.seed(self.args.seed)
        self.train_env.observation_space.seed(self.args.seed)
        self.eval_env.reset(seed = self.args.seed)
        self.eval_env.action_space.seed(self.args.seed)
        self.eval_env.observation_space.seed(self.args.seed)
        self.robust_env.reset(seed=self.args.seed)
        self.robust_env.action_space.seed(self.args.seed)
        self.robust_env.observation_space.seed(self.args.seed)

    def train(self):
        state, done, episode_return, episode_length = self.train_env.reset(), False, 0., 0
        for _ in range(1, self.args.num_train_steps+1):  
            if self._global_step <= self.args.learning_starts:
                action = self.train_env.action_space.sample()
            else:
                action = self.agent.get_action(state)

            next_state, reward, done, _ = self.train_env.step(action)
            episode_return += reward
            episode_length += 1

            if done and episode_length == self.train_env._max_episode_steps:
                true_done = False 
            else:
                true_done = done

            self.agent.env_buffer.push((state, action, reward, next_state, true_done))

            if len(self.agent.env_buffer) > self.args.batch_size and self._global_step > self.args.learning_starts:
                start = time.time()
                self.agent.update(self._global_step)
                duration_step = time.time()-start
            
            if (self._global_step)%1000==0:
                self.eval_robustness()

            if (self._global_step+1)%self.args.eval_episode_interval==0:
                self.eval()

            if self._global_step%self.args.save_snapshot_interval==0:
                self.save_snapshot()

            self._global_step += 1
            if done:
                self._global_episode += 1
                print("Episode: {}, total numsteps: {}, return: {}".format(self._global_episode, self._global_step, round(episode_return, 2)))
                episode_metrics = {}
                if self._global_step > self.args.learning_starts:
                    episode_metrics['duration_step'] = duration_step
                episode_metrics['episodic_length'] = episode_length
                episode_metrics['episodic_return'] = episode_return
                episode_metrics['env_buffer_length'] = len(self.agent.env_buffer)

                wandb.log(episode_metrics, step=self._global_step)
                state, done, episode_return, episode_length = self.train_env.reset(), False, 0., 0
            else:
                state = next_state
                
        self.train_env.close()
 
    def eval(self):
        steps, returns = 0, 0
        for e in range(self.args.num_eval_episodes):
            done = False 
            state = self.eval_env.reset()
            while not done:
                with torch.no_grad():
                    action = self.agent.get_action(state, True)
                next_state, reward, done , _ = self.eval_env.step(action)
                self.eval_env.render()
                time.sleep(0.1)
                returns += reward
                steps += 1
                state = next_state

        if returns/self.args.num_eval_episodes >= self._best_eval_returns:
            self.save_snapshot(best=True)
            self._best_eval_returns = returns/self.args.num_eval_episodes

        eval_metrics = {}
        eval_metrics['eval_episodic_return'] = returns/self.args.num_eval_episodes
        eval_metrics['eval_episodic_length'] = steps/self.args.num_eval_episodes
        wandb.log(eval_metrics, step = self._global_step)

    def eval_robustness(self):
        steps, returns = 0, 0

        for e in range(2):
            done = False 
            state = self.robust_env.reset()
            while not done:
                r = random.uniform(0.0, 1.0)
                if r>0.5:
                    state[0] += 2*(np.random.rand()-0.5) 
                with torch.no_grad():
                    action = self.agent.get_action(state, True)
                next_state, reward, done , _ = self.robust_env.step(action)
                returns += reward
                steps += 1
                state = next_state

        robust_metrics = {}
        robust_metrics['robust_episodic_return'] = returns/2
        robust_metrics['robust_episodic_length'] = steps/2
        wandb.log(robust_metrics, step = self._global_step)

    def save_snapshot(self, best=False):
        keys_to_save = ['agent', '_global_step', '_global_episode']
        if best:
            snapshot = Path(self.checkpoint_path) / 'best.pt'
        else:
            snapshot = Path(self.checkpoint_path) / Path(str(self._global_step)+'.pt')
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, iter, best=False):
        if best:
            snapshot = Path(self.checkpoint_path) / 'best.pt'
        else:
            snapshot = Path(self.checkpoint_path) / Path(str(iter)+'.pt')
            
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        
    def load_best(self):
        snapshot = Path('best.pt')
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        
def main():

    with open("mujoco.yaml", 'r') as stream:
        mujoco_config = yaml.safe_load(stream)
    args = parse_args(mujoco_config['rmbrl_params'])
    
    #with wandb.init(project=args.agent, entity='raj19', group=args.env_name, config=args.__dict__):
    #wandb.run.name = args.env_name+'_'+str(args.seed)
    workspace = MujocoWorkspace(args)
    workspace.load_best()
    workspace.eval()
    #workspace.train()

if __name__ == '__main__':
    main()