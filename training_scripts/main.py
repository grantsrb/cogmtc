import torch
import cogmtc.training
from cogmtc.utils.training import run_training
import torch.multiprocessing as mp

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    run_training(cogmtc.training.train)

