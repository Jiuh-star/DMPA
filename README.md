# Dual Projection Model Poisoning Attack

This is code implementation for `Dual Projection: Persistent Model Poisoning Attack against Fairness and Robustness in Federated Learning` and other model poisoning attacks, see code for details.

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
python3 system gendata -c data-config.json5
```
2. (Optional) Check your data distribution.
```bash
python3 system plot -d ${Path-To-FL-Dataset-File}
```
3. Start the federated learning system.
```bash
python3 system run -c config.json5
```