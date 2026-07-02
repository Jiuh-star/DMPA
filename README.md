> [!NOTE]
> Following [this draft](https://www.ietf.org/archive/id/draft-abaris-aicdh-00.html), we disclosure the presence and degree of vibe coding in the repo.
> 
> `AI-Disclosure: none`

# Durable Model Poisoning Attack

This is partial code implementation for `DMPA: Durable Model Poisoning Attack against Fairness and Robustness in Efficient Federated Learning System` and other model poisoning attacks in `FedAvg`, see code for details. Note that we will update the full implementation after the paper is accepted.

## Project Structure

```bash
./
├─ attackers  # Code for MPAs
├─ flcore  # A simple and lightweight federated learning framework
└─ system  # Experiment environment, we implemented FedAvg here
```

## Requirements
`pytorch~=2.0.0`, `torchvision` (model and dataset), `json5` (configuration parser), `typer` (CLI), `rich` (logging and colored stdout), `python-box` (prefered configuration container), `more-itertools` (iteration utilities), `torchmetrics` (metrics), `tensorboard` (dashboard).


## How to Start

1. Generate data for the federated learning system.
```bash
python3 -m system gendata -c configs/data/cifar10-cnn.json5
```
2. (Optional) Check your data distribution.
```bash
python3 -m system plot -d configs/data/cifar10-cnn.fl
```
3. Start the federated learning system.
```bash
python3 -m system run -c configs/cifar10-cnn-mkrum/dmpa.json5
```
4. Check experimental result with TensorBoard in output folder.
```bash
tensorboard --logdir output/MultiKrum/CIFAR10-CNN/DMPA/
```

## BibTex
```bibtex
@article{jiang2026dmpa,
  title={DMPA: Durable Model Poisoning Attack against Fairness and Robustness in Efficient Federated Learning Systems},
  author={Jiang, Jionghui and Hao, Fengrui and Gu, Tianlong and Wang, Ke and Liu, Xiaoli and Wen, Zhangbin},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2026},
  publisher={IEEE}
}
```
