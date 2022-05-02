"""
This script takes a metaranges file and equally distributes each combo
of hyperranges (specified within the metaranges) to a tmux session
running on its own gpu.

    $ python3 distr.py main.py metaranges.json

The meta ranges should have the following structure within a .json:

{
    "devices": [0,1,2],
    "hyperparams": "path/to/hyperparams.json",
    "hyperranges": "path/to/hyperranges.json"
}

or

{
    "devices": [0,1,2],
    "split_keys": ["use_count_words"],
    "hyperparams": "path/to/hyperparams.json",
    "hyperranges": "path/to/hyperranges.json"
}
"""
import sys
import os
from cogmtc.utils.utils import load_json, save_json
from cogmtc.utils.training import fill_hyper_q
from datetime import datetime
from collections import deque
import math

# os.system("tmux new -s tes")
# tmux new-session -d -s \"myTempSession\" /opt/my_script.sh

def distr_ranges(script, meta, rng_paths):
    exp_name = load_json(meta["hyperparams"])["exp_name"]
    stdout_folder = "./tmux_logs/"
    if not os.path.exists(stdout_folder):
        os.mkdir(stdout_folder)

    tmux_sesh = "tmux new -d -s"
    exe = "python3 {}".format(script)
    for rng_path, device in zip(rng_paths, meta["devices"]):
        cuda = "export CUDA_VISIBLE_DEVICES=" + str(device)
        deterministic = "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
        sesh_name = "{}{}".format(exp_name[:4],device)
        timestamp = str(datetime.now()).replace(" ", "_")
        timestamp = timestamp.split(".")[0].replace(":",".")
        fname = sesh_name+"_"+timestamp+".txt"
        log_file = os.path.join(stdout_folder, fname)
        command = "{} \"{}\" \'{}; {}; {} {} {}\'".format(
            tmux_sesh,
            sesh_name,
            cuda,
            deterministic,
            exe,
            meta["hyperparams"],
            rng_path
        )
        print(command)
        os.system(command)

def split_ranges(meta):
    """
    Takes a hyperranges file and splits the ranges on the split_key into
    multiple different ranges files. One for each cuda device.

    Args:
        meta: dict
            "hyperparams": str
                path to a hyperparams file
            "hyperranges": str
                path to a hyperranges file
            "key_order": str
                the key that should be distributed among devices
            "devices": list of int
                the potential cuda device indices to train on
    Returns:
        rng_paths: list of str
            a list of paths to the new hyperranges files
    """
    ranges = load_json(meta["hyperranges"])

    # Save to folder that we found the ranges
    # Each ranges is saved as exp_name{cuda_device}.json
    save_path = os.path.abspath(meta["hyperranges"]).split("/")
    save_path[-1] = load_json(meta["hyperparams"])["exp_name"]
    save_path = "/".join(save_path)

    devices = meta["devices"]

    # Get queue of hyperparameter combinations divided by importance
    key_importances = meta["key_order"]
    for k in ranges.keys()-set(meta["key_order"]):
        key_importances.append(k)
    hyper_q = deque()
    hyper_q = fill_hyper_q({},ranges,key_importances,hyper_q,idx=0)

    # Divide up hyperranges equally amongst GPUs
    n_combos = math.ceil(len(hyper_q)/len(devices))
    rng_paths = []
    range_dict = {i:None for i in devices}
    for i,d in enumerate(devices):
        combos = None
        for combo in range(min(n_combos,len(hyper_q))):
            if combos is None:
                combos = {k:[v] for k,v in hyper_q.pop().items()}
            else:
                for k,v in hyper_q.pop().items():
                    combos[k].append(v)
        if combos is not None:
            del combos["search_keys"]
            range_dict[d] = { "combos": combos }
        else:
            del range_dict[d]
            del devices[i]
            print("Leaving device", d, "unused!!")

    # Save hyperranges to json files
    for device in devices:
        path = save_path+"{}.json".format(device)
        rng_paths.append(path)
        save_json(range_dict[device], path)
    return rng_paths

if __name__ == "__main__":

    meta = load_json(sys.argv[2])
    rng_paths = split_ranges(meta)
    #distr_ranges(sys.argv[1], meta, rng_paths)
