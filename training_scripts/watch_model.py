import time
import torch
import numpy as np
import cogmtc
import sys
from cogmtc.models import *
import matplotlib.pyplot as plt

"""
Must argue the path to a model folder for viewing. This script
automatically selects the best model from the training.

$ python3 watch_model.py exp_name/model_folder/
"""
render = False
verbose = True
targ_range = [15,22]

if __name__ == "__main__":
    model_folder = sys.argv[1]
    checkpt = cogmtc.utils.save_io.load_checkpoint(
        model_folder,
        use_best=False
    )
    hyps = checkpt["hyps"]
    hyps["n_eval_steps"] = 1000
    hyps["seed"] = int(time.time())
    hyps["render"] = render
    hyps["val_targ_range"] = targ_range
    model = cogmtc.utils.save_io.load_model(checkpt)
    model.cuda()
    model.eval()
    model.reset()
    val_runner = cogmtc.experience.ValidationRunner(hyps)
    val_runner.phase = 2
    eval_eps = 3
    for env_type in hyps["env_types"]:
        print("EnvType:", env_type)
        val_runner.oracle = val_runner.oracles[env_type]
        data = val_runner.collect_data(
            model, n_targs=None, env_type=env_type, verbose=verbose
        )
