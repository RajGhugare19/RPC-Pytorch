# RPC-Pytorch
Pytorch replication of the paper : [Robust Predictable Control](https://arxiv.org/abs/2109.03214)

# Run code

Run the original rpc agent with a KL constraint of 10 bits / timestep and evaluate with a noise factor of 3.

  ```
  python train.py --agent 'RPC' --kl_constraint 10 --noise_factor 3
  ``` 
 
Run the rpc agent with a recurrent prior with a KL constraint of 1 bits / timestep and evaluate with a noise factor of 2.
  
  ```
  python train.py --agent 'RRPC' --kl_constraint 10 --noise_factor 2
  ``` 
