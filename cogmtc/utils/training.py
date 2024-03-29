import subprocess
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import PoissonNLLLoss, MSELoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR
import time
from tqdm import tqdm
import math
from queue import Queue
from collections import deque
import psutil
import json
import cogmtc.utils.save_io as io
import cogmtc.utils.utils as utils
import select
import shutil
import os
import torch.multiprocessing as mp
from datetime import datetime

def get_resume_checkpt(hyps, in_place=False, verbose=True):
    """
    This function cleans up the code to resume from a particular
    checkpoint or folder.

    Args:
        hyps: dict
            dictionary of hyperparameters
            keys: str
                "resume_folder": str
                    must be a key present in hyps for this function to act.
                "ignore_keys": list of str
                    an optional key used to enumerate keys to be ignored
                    when loading the old hyperparameter set
            vals: varies
        in_place: bool
            if true, changes the argued hyperparameters in place.
            otherwise has no effect on the argued hyperparameters
    Returns:
        checkpt: dict or None
            a loaded checkpoint dict of the model that will be resumed
            from. None if the checkpoint does not exist
        hyps: dict
            the modified hyperparameters. This does not reference the
            argued hyps object. But will be a deep copy of the argued
            hyps if `in_place` is true.
    """
    # Ignore keys are used to override the hyperparams json associated
    # with the loaded training. In otherwords, use the ignore keys
    # to specify new hyperparameters in the current training instead
    # of the hyperparameters from the training that is being loaded.
    ignore_keys = [
        'n_epochs',
        "lang_epochs",
        "actn_epochs",
        'rank',
        "n_eval_steps",
        "n_eval_eps",
        "save_root",
        "resume_folder",
    ]
    ignore_keys = utils.try_key(hyps,'ignore_keys',ignore_keys)
    resume_folder = utils.try_key(hyps,'resume_folder', None)
    if not in_place: 
        hyps = {**hyps}
    if resume_folder is not None and resume_folder != "":
        checkpt = io.load_checkpoint(resume_folder)
        # Check if argued number of epochs exceeds the total epochs
        # of the to-be-resumed training
        if verbose and training_exceeds_epochs(hyps, checkpt):
            print("Could not resume due to epoch count")
            print("Performing fresh training")
        else:
            if verbose:
                print("Loading hyps from", resume_folder)
            temp_hyps = checkpt['hyps']
            for k,v in temp_hyps.items():
                if k not in ignore_keys:
                    hyps[k] = v
            hyps['seed'] += 54321 # For fresh data
            s = " Restarted training from epoch "+str(checkpt['epoch'])
            hyps['description'] = utils.try_key(
                hyps,
                "description",
                ""
            )
            hyps['description'] += s
            hyps['ignore_keys'] = ignore_keys
            hyps["save_folder"] = resume_folder
            return checkpt, hyps
    return None, hyps

def training_exceeds_epochs(hyps, checkpt):
    """
    Helper function to deterimine if the training to be resumed has
    already exceeded the number of epochs argued in the resumed
    training.

    Args:
        hyps: dict
        checkpt: dict
    """
    n_epochs = -1
    if "phase" in checkpt and checkpt["phase"]==0:
        n_epochs = hyps["lang_epochs"]
    elif "phase" in checkpt and checkpt["phase"] > 0:
        n_epochs = hyps["actn_epochs"]
    elif "n_epochs" in hyps: n_epochs = hyps["n_epochs"]
    return (checkpt['epoch']>=(n_epochs-1) or n_epochs == -1)

def get_exp_num(exp_folder, exp_name, offset=0):
    """
    Finds the next open experiment id number by searching through the
    existing experiment numbers in the folder.

    If an offset is argued, it is impossible to have an exp_num that is
    less than the value of the offset. The returned exp_num will be
    the next available experiment number starting with the value of the
    offset.

    Args:
        exp_folder: str
            path to the main experiment folder that contains the model
            folders (should not include the experiment name as the final
            directory)
        exp_name: str
            the name of the experiment
        offset: int
            a number to offset the experiment numbers by.

    Returns:
        exp_num: int
    """
    name_splt = exp_name.split("_")
    namedex = 1
    if len(name_splt) > 1:
        namedex = len(name_splt)
    exp_folder = os.path.expanduser(exp_folder)
    _, dirs, _ = next(os.walk(exp_folder))
    exp_nums = set()
    for d in dirs:
        splt = d.split("_")
        if len(splt) >= 2:
            num = None
            for i in range(len(splt)):
                try:
                    num = int(splt[i])
                    break
                except:
                    pass
            if namedex > 1 and i > 1:
                name = "_".join(splt[:namedex])
            else: name = splt[0]
            if name == exp_name and num is not None:
                exp_nums.add(num)
    for i in range(len(exp_nums)):
        if i+offset not in exp_nums:
            return i+offset
    return len(exp_nums) + offset

def get_save_folder(hyps):
    """
    Creates the save name for the model.

    hyps: dict
        keys:
            main_path: str
                the leading path to the save_folder
            exp_name: str
                the root experiment name
            exp_num: int
                the experiment id number
            search_keys: str
                the identifying keys for this hyperparameter search
    """
    save_folder = "{}/{}_{}".format(hyps['main_path'],
                                    hyps['exp_name'],
                                    hyps['exp_num'])
    save_folder += prep_search_keys(hyps["search_keys"])
    return save_folder

def get_git_revision_hash():
    """
    Finds the current git hash
    """
    return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()

def record_session(hyps, model):
    """
    Writes important parameters to file. If 'resume_folder' is an entry
    in the hyps dict, then the txt file is appended to instead of being
    overwritten.

    hyps: dict
        dict of relevant hyperparameters
    model: torch nn.Module
        the model to be trained
    """
    git_hash = hyps["git_hash"]
    sf = hyps['save_folder']
    if not os.path.exists(sf):
        os.mkdir(sf)
    h = "hyperparams"
    mode = "a" if "resume_folder" in hyps else "w"
    with open(os.path.join(sf,h+".txt"),mode) as f:
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write("\nGit Hash: {}".format(git_hash))
        f.write("\n"+str(model)+'\n')
        for k in sorted(hyps.keys()):
            f.write(str(k) + ": " + str(hyps[k]) + "\n")
    temp_hyps = dict()
    keys = list(hyps.keys())
    temp_hyps = {k:v for k,v in hyps.items()}
    for k in keys:
        if type(hyps[k]) == type(np.array([])):
            del temp_hyps[k]
        elif type(hyps[k]) == type(set()): temp_hyps[k] = list(hyps[k])
    utils.save_json(temp_hyps, os.path.join(sf,h+".json"))

def prep_search_keys(s):
    """
    Removes unwanted characters from the search keys string.
    """
    return s.replace(
            " ", ""
        ).replace(
            "[", ""
        ).replace(
            "]", ""
        ).replace(
            "\'", ""
        ).replace(
            ",", ""
        ).replace(
            "/", ""
        )
    
def fill_hyper_q(hyps, hyp_ranges, keys, hyper_q, idx=0):
    """
    Recursive function to load each of the hyperparameter combinations
    onto a queue.

    hyps - dict of hyperparameters created by a HyperParameters object
        type: dict
        keys: name of hyperparameter
        values: value of hyperparameter
    hyp_ranges - dict of lists
        these are the ranges that will change the hyperparameters for
        each search
        type: dict
        keys: name of hyperparameters to be searched over
        values: list or dict of lists of equal length
            if a list is given, it should be a list of values to search
            over for the hyperparameter specified by the corresponding
            key. If a dict of lists is given, the name of the key is
            ignored and the items in the lists are paired together for
            a hyperparameter combination corresponding to their
            respective keys. i.e.

            {
                "param0": [foo1],
                "combos": {
                    "param1":[p1,p2],
                    "param2":[v1,v2]
                }
            }

            will result in the folowing 2 hyperparameter search values:

            {"param0": foo1, "param1": p1, "param2": v1}
            and
            {"param0": foo1, "param1": p2, "param2": v2}
    keys - keys of the hyperparameters to be searched over. Used to
        specify order of keys to search
    hyper_q - deque to hold all parameter sets
    idx - the index of the current key to be searched over

    Returns:
        hyper_q: deque of dicts `hyps`
    """
    # Base call, saves the hyperparameter combination
    if idx >= len(keys):
        # Load q
        hyps['search_keys'] = ""
        for k in keys:
            if isinstance(hyp_ranges[k],dict):
                for rk in hyp_ranges[k].keys():
                    s = prep_search_keys(str(hyps[rk]))
                    hyps['search_keys'] += "_" + str(rk)+s
            else:
                s = prep_search_keys(str(hyps[k]))
                hyps['search_keys'] += "_" + str(k)+s
        hyper_q.append({**hyps})

    # Non-base call. Sets a hyperparameter to a new search value and
    # passes down the dict.
    else:
        key = keys[idx]
        # Allows us to specify combinations of hyperparameters
        if isinstance(hyp_ranges[key],dict):
            rkeys = list(hyp_ranges[key].keys())
            for i in range(len(hyp_ranges[key][rkeys[0]])):
                for rkey in rkeys:
                    hyps[rkey] = hyp_ranges[key][rkey][i]
                hyper_q = fill_hyper_q(hyps, hyp_ranges, keys, hyper_q,
                                                               idx+1)
        else:
            for param in hyp_ranges[key]:
                hyps[key] = param
                hyper_q = fill_hyper_q(hyps, hyp_ranges, keys, hyper_q,
                                                               idx+1)
    return hyper_q

def make_hyper_range(low, high, range_len, method="log"):
    """
    Creates a list of length range_len that is a range between two
    values. The method dictates the spacing between the values.

    low: float
        the lowest value in the range

    """
    if method.lower() == "random":
        param_vals = np.random.random(low, high+1e-5, size=range_len)
    elif method.lower() == "uniform":
        step = (high-low)/(range_len-1)
        param_vals = np.arange(low, high+1e-5, step=step)
    else:
        range_low = np.log(low)/np.log(10)
        range_high = np.log(high)/np.log(10)
        step = (range_high-range_low)/(range_len-1)
        arange = np.arange(range_low, range_high+1e-5, step=step)
        param_vals = 10**arange
    param_vals = [float(param_val) for param_val in param_vals]
    return param_vals

def hyper_search(hyps, hyp_ranges, train_fxn):
    """
    The top level function to create hyperparameter combinations and
    perform trainings.

    hyps: dict
        the initial hyperparameter dict
        keys: str
        vals: values for the hyperparameters specified by the keys
    hyp_ranges: dict
        these are the ranges that will change the hyperparameters for
        each search. A unique training is performed for every
        possible combination of the listed values for each key
        keys: str
        vals: lists of values for the hyperparameters specified by the
              keys
    train_fxn: function
        args:
            hyps: dict
            verbose: bool
        a function that performs the desired training given the argued
        hyperparams
    """
    starttime = time.time()
    # Make results file
    main_path = hyps['exp_name']
    if "save_root" in hyps:
        hyps['save_root'] = os.path.expanduser(hyps['save_root'])
        if not os.path.exists(hyps['save_root']):
            os.mkdir(hyps['save_root'])
        main_path = os.path.join(hyps['save_root'], main_path)
    if not os.path.exists(main_path):
        os.mkdir(main_path)

    hyps['multi_gpu'] = utils.try_key(hyps,'multi_gpu',False)
    if hyps['multi_gpu']:
        hyps["n_gpus"] = torch.cuda.device_count()
        hyps['world_size'] = hyps['n_nodes']*hyps['n_gpus']
        os.environ['MASTER_ADDR'] = '127.0.0.1'     
        os.environ['MASTER_PORT'] = '8021'
    hyps['main_path'] = main_path
    results_file = os.path.join(main_path, "results.txt")
    with open(results_file,'a') as f:
        f.write("Hyperparameters:\n")
        for k in hyps.keys():
            if k not in hyp_ranges:
                f.write(str(k) + ": " + str(hyps[k]) + '\n')
        f.write("\nHyperranges:\n")
        for k in hyp_ranges.keys():
            if isinstance(hyp_ranges[k],dict):
                s = str(k)+":\n"
                for rk in hyp_ranges[k].keys():
                    rs = ",".join([str(v) for v in hyp_ranges[k][rk]])
                    s += "  "+str(rk) + ": [" + rs +']\n'
            else:
                rs = ",".join([str(v) for v in hyp_ranges[k]])
                s = str(k) + ": [" + rs +']\n'
            f.write(s)
        f.write('\n')

    hyper_q = deque()
    hyper_q = fill_hyper_q(hyps, hyp_ranges, list(hyp_ranges.keys()),
                                                      hyper_q, idx=0)
    total_searches = len(hyper_q)
    print("n_searches:", total_searches)

    result_count = 0
    print("Starting Hyperloop")
    while not len(hyper_q)==0:
        print("\n\nSearches left:", len(hyper_q),"-- Running Time:",
                                             time.time()-starttime)
        hyps = hyper_q.popleft()

        verbose = True
        if hyps['multi_gpu']:
            mp.spawn(train_fxn, nprocs=hyps['n_gpus'],
                                args=(hyps,verbose))
        else:
            train_fxn(0, hyps=hyps, verbose=verbose)

def run_training(train_fxn):
    """
    This function extracts the hyperparams and hyperranges from the
    command line arguments and asks the user if they would like to
    proceed with the training and/or overwrite the previous save
    folder.

    train_fxn: function
        this the training function that will carry out the training
        args:
            hyps: dict
            verbose: bool
    """
    hyps = utils.load_json(sys.argv[1])
    print()
    print("Using hyperparams file:", sys.argv[1])
    if len(sys.argv) < 3:
        ranges = {"lr": [hyps['lr']]}
    else:
        ranges = utils.load_json(sys.argv[2])
        print("Using hyperranges file:", sys.argv[2])
    print()

    keys = sorted(list(hyps.keys()))
    hyps_str = ""
    for k in keys:
        if k not in ranges:
            hyps_str += "{}: {}\n".format(k,hyps[k])
    print("Hyperparameters:")
    print(hyps_str)
    print("\nSearching over:")
    print("\n".join(["{}: {}".format(k,v) for k,v in ranges.items()]))

    main_path = hyps['exp_name']
    if "save_root" in hyps:
        hyps['save_root'] = os.path.expanduser(hyps['save_root'])
        if not os.path.exists(hyps['save_root']):
            os.mkdir(hyps['save_root'])
        main_path = os.path.join(hyps['save_root'], main_path)
    sleep_time = 8
    if os.path.exists(main_path):
        dirs = io.get_model_folders(main_path)
        if len(dirs) > 0:
            s = "Overwrite last folder {}? (No/yes)".format(dirs[-1])
            print(s)
            i,_,_ = select.select([sys.stdin], [],[],sleep_time)
            if i and "y" in sys.stdin.readline().strip().lower():
                print("Are you sure?? This will delete the data (Y/n)")
                i,_,_ = select.select([sys.stdin], [],[],sleep_time)
                if i and "n" not in sys.stdin.readline().strip().lower():
                    path = os.path.join(main_path, dirs[-1])
                    shutil.rmtree(path, ignore_errors=True)
        else:
            s = "You have {} seconds to cancel experiment name {}:"
            print(s.format(sleep_time, hyps['exp_name']))
            i,_,_ = select.select([sys.stdin], [],[],sleep_time)
    else:
        s = "You have {} seconds to cancel experiment name {}:"
        print(s.format(sleep_time, hyps['exp_name']))
        i,_,_ = select.select([sys.stdin], [],[],sleep_time)
    print()

    keys = list(ranges.keys())
    start_time = time.time()
    hyper_search(hyps, ranges, train_fxn)
    print("Total Execution Time:", time.time() - start_time)

