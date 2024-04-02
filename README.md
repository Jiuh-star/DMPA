# Durable Model Poisoning Attack

This is code implementation for `DMPA: Durable Model Poisoning Attack against Fairness and Robustness in Federated Learning` and other model poisoning attacks, see code for details.

## Project Structure

```bash
./
├─ attackers  # Code for MPAs
├─ flcore  # A simple and lightweight federated learning framework
└─ system  # Experiment environment, we implemented FedAvg here
```



## How to Start

1. Generate data for the federated learning system.
```bash
python3 system gendata -c configs/data/cifar10-cnn.json5
```
2. (Optional) Check your data distribution.
```bash
python3 system plot -d configs/data/cifar10-cnn.fl
```
3. Start the federated learning system.
```bash
python3 system run -c configs/cifar10-cnn-mkrum/dmpa.json5
```
4. Check experimental result with TensorBoard in output folder.
```bash
tensorboard --logdir output/MultiKrum/CIFAR10-CNN/DMPA/
```