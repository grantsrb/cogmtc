"""
Use this script to collect rollouts from the final models in each of
the argued folders

 $ python3 eval_models.py path/to/folder/ path/to/model

optionally can argue a csv file to save to. Simply add a valid path
with .csv as the extension. Also can argue "best" or "bests" to use
the model's best checkpoint files.
 
 $ python3 eval_models.py bests valid_path.csv path/to/folder/ path/to/model

"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import os
import cogmtc
import cogmtc.utils.utils as utls
import cogmtc.utils.save_io as io
from   cogmtc.models import *
import scipy.stats as ss
from tqdm import tqdm
import collections
import gym
import gordongames
import gordoncont
import sys
import time


folder_args = sys.argv[1:]
main_csv = "model_eval.csv"
model_folders = []
use_best = False
for folder in folder_args:
    if folder == "best" or folder == "bests":
        use_best = True
    elif ".csv" in folder:
        main_csv = folder
    else:
        folders = io.get_model_folders(folder, incl_full_path=True)
        if len(folders) > 0: model_folders = [*model_folders, *folders]

# Filter unwanted models
keeps = {
}
drops = {
}

keep_idxs = []
for i,folder in enumerate(model_folders):
    hyps = io.get_hyps( folder )
    keep = True
    for k in keeps:
        if k not in hyps or hyps[k] not in keeps[k]:
            keep = False
            break
    if not keep: continue
    drop = False
    for d in drops:
        if d not in hyps or hyps[d] in drops[d]:
                drop = True
                break
    if not drop:
        keep_idxs.append(i)

folders = model_folders
model_folders = []
for idx in keep_idxs:
    model_folders.append(folders[idx])

print("Model Folders:")
for i,folder in enumerate(model_folders):
    print(i,folder)


if os.path.exists(main_csv):
    main_df = pd.read_csv(main_csv, sep="!")
else:
    main_df = pd.DataFrame()
print()
for i,folder in enumerate(model_folders):
    start_time = time.time()
    if "model_folder" in main_df.columns and\
            folder in set(main_df["model_folder"]):
        print("Entries already exists for", folder)
        continue
    print("Analyzing Folder {}:".format(i), folder)

    hyps = io.load_hyps(folder)

    # Collect Eval Data for Each Folder
    datas = {
        "n_targs": [],
        "n_items": [],
        "lang_preds": [],
        "actn_targs": [],
        "lang_targs": [],
        "is_animating": [],
        "dones": [],
    }
    meta_datas = {
        "env_type": [],
        "model_folder": [],
        "count_type": [],
    }
    hyp_datas = {k: [] for k in hyps.keys()}
    
    count_type = hyps["use_count_words"]
    if count_type==1 and hyps["skip_first_phase"] and hyps["second_phase"]==1:
        count_type = -1

    hyps["val_targ_range"] = [1,12]
    val_runner = cogmtc.experience.ValidationRunner(hyps, phase=2)
    model = io.load_model( folder, use_best=use_best )
    model.cuda()
    model.eval()
    model.reset(1)
    for env_type in hyps["env_types"]:
        print("EnvType:", env_type)
        val_runner.oracle = val_runner.oracles[env_type]
        with torch.no_grad():
            for i in tqdm(range(1,hyps["val_targ_range"][-1]+1)):
                data = val_runner.collect_data(
                    model,
                    render=False,
                    env_type=env_type,
                    n_targs=i,
                    to_cpu=True,
                    n_eps=50)
                # Collect data into dicts to later become dataframe
                data["lang_targs"] = torch.zeros_like(data["dones"])
                if count_type >= 0:
                    data["lang_targs"] = utls.get_lang_labels(
                        data["n_items"],
                        data["n_targs"],
                        model.lang_size-1,
                        use_count_words=count_type
                    )
                    if count_type == 5:
                        base = hyps["numeral_base"]
                        fxn = utls.convert_numeral_array_to_numbers
                        data["lang_targs"] = fxn(
                            data["lang_targs"], base
                        )
                data["lang_targs"] = data["lang_targs"].data.numpy()
                for k in datas.keys():
                    if k == "lang_preds":
                        # data[k] has shape (S, L)
                        if count_type == 5:
                            base = hyps["numeral_base"]
                            s = data[k].shape
                            new_shape = (*s[:-1],-1,base+1)
                            data[k] = data[k].reshape(new_shape)
                            data[k] = torch.argmax(data[k], dim=-1)
                            fxn = utls.convert_numeral_array_to_numbers
                            data[k] = fxn( data[k], base )
                        else:
                            data[k] = torch.argmax(data[k],axis=-1)
                    datas[k].append(data[k])
                idx = data["dones"]==1
                temp = (data["n_targs"][idx]==data["n_items"][idx])
                print("Acc:", temp.float().mean().item())
                for k,v in zip(
                    ["env_type", "model_folder", "count_type"],
                    [env_type, folder, count_type]
                ):
                    meta_datas[k] = [
                        *meta_datas[k],
                        *[v for _ in range(len(data["dones"]))]
                    ]
                for k in hyp_datas:
                    hyp_datas[k] = [
                        *hyp_datas[k],
                        *[hyps[k] for _ in range(len(data["dones"]))]
                    ]
        
    for k in datas.keys():
        if k == "lang_preds":  
            datas[k] = torch.cat(datas[k], axis=1)
            datas[k] = datas[k].squeeze().cpu().detach().data.numpy()
        elif k == "actn_preds":
            datas[k] = torch.cat(datas[k], axis=0)
            datas[k] = datas[k].cpu().detach().data.numpy().squeeze()
        else: datas[k] = np.concatenate(datas[k], axis=0)

    df = {**datas, **meta_datas, **hyp_datas}
    df = pd.DataFrame(df)
    main_df = main_df.append(df, sort=True)
    main_df.to_csv(main_csv, sep="!", index=False, header=True, mode="w")
    print("Total Eval Time:", time.time()-start_time)
    print()
    print()
    print()
