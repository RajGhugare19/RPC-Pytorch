# RPC-Pytorch
Pytorch replication of the paper : [Robust Predictable Control](https://arxiv.org/abs/2109.03214)

# Run code

Run the original rpc agent with a KL constraint of 10 bits/t and evaluate with a noise factor of 3.

  ```
  python rpc.py --kl_constraint 10 --noise_factor 3
  ``` 
 
Run the rpc agent with a recurrent prior with a KL constraint of 10 bits/t and evaluate with a noise factor of 2.
  
  ```
  python rrpc.py --kl_constraint 10 --noise_factor 2
  ``` 
