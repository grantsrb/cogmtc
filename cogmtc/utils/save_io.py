import torch
import pickle
import cogmtc.models
import cogmtc.utils.utils as cuu
import os
import math

BEST_CHECKPT_NAME = "best_checkpt_0.pt.best"

def save_checkpt(save_dict, save_name, epoch, ext=".pt",
                                       del_prev_sd=True,
                                       best=False):
    """
    Saves a dictionary that contains a statedict

    save_dict: dict
        a dictionary containing all the things you want to save
    save_name: str
        the path to save the dict to.
    epoch: int
        an integer to be associated with this checkpoint
    ext: str
        the extension of the file
    del_prev_sd: bool
        if true, the state_dict of the previous checkpoint will be
        deleted
    best: bool
        if true, additionally saves this checkpoint as the best
        checkpoint under the filename set by BEST_CHECKPT_NAME
    """
    if del_prev_sd and epoch is not None:
        prev_path = "{}_{}{}".format(save_name,epoch-1,ext)
        prev_path = os.path.abspath(os.path.expanduser(prev_path))
        if os.path.exists(prev_path):
            delete_sds(prev_path)
        elif epoch != 0:
            print("Failed to find previous checkpoint", prev_path)
    if epoch is None: epoch = 0
    path = "{}_{}{}".format(save_name,epoch,ext)
    path = os.path.abspath(os.path.expanduser(path))
    torch.save(save_dict, path)
    if best:
        if "/" not in save_name:
            folder = os.path.join("./")
        else:
            folder = save_name.split("/")[:-1]
            folder = os.path.join(*folder)
            if save_name[0] == "/": folder = "/" + folder
        save_best_checkpt(save_dict, folder)

def delete_sds(checkpt_path):
    """
    Deletes the state_dicts from the argued checkpt path.

    Args:
        checkpt_path: str
            the full path to the checkpoint
    """
    if not os.path.exists(checkpt_path): return
    checkpt = load_checkpoint(checkpt_path)
    keys = list(checkpt.keys())
    for key in keys:
        if "state_dict" in key or "optim_dic" in key:
            del checkpt[key]
    torch.save(checkpt, checkpt_path)

def save_best_checkpt(save_dict, folder):
    """
    Saves the checkpoint under the name set in BEST_CHECKPT_PATH to the
    argued folder

    save_dict: dict
        a dictionary containing all the things you want to save
    folder: str
        the path to the folder to save the dict to.
    """
    path = os.path.join(folder,BEST_CHECKPT_NAME)
    path = os.path.abspath(path)
    torch.save(save_dict,path)

def get_checkpoints(folder, checkpt_exts={'p', 'pt', 'pth'}):
    """
    Returns all .p, .pt, and .pth file names contained within the
    folder. They're sorted by their epoch.

    BEST_CHECKPT_PATH is not included in this list. It is excluded using
    the assumption that it has the extension ".best"

    folder: str
        path to the folder of interest
    checkpt_exts: set of str
        a set of checkpoint extensions to include in the checkpt search.

    Returns:
        checkpts: list of str
            the full paths to the checkpoints contained in the folder
    """
    folder = os.path.expanduser(folder)
    assert os.path.isdir(folder)
    checkpts = []
    for f in os.listdir(folder):
        splt = f.split(".")
        if len(splt) > 1 and splt[-1] in checkpt_exts:
            path = os.path.join(folder,f)
            checkpts.append(path)
    def sort_key(x):
        phase = -1
        splt = x.split("phase")
        if len(splt) >= 1:
            try:
                phase = int(splt[-1].split("_")[0])
            except:
                pass
        epoch = int(x.split(".")[-2].split("_")[-1])
        return (phase, epoch)
    checkpts = sorted(checkpts, key=sort_key)
    return checkpts

def get_phase_checkpoints(folder,phase,checkpt_exts={'p', 'pt', 'pth'}):
    """
    Finds and returns all checkpoints of the phase that is specified.

    Args:
        folder: str
            path to the folder of interest
        phase: int or str
            the phase of the training that you would like to focus the
            search on.
        checkpt_exts: set of str
            a set of checkpoint extensions to include in the checkpt search.
    Returns:
        checkpts: list of str
            the full paths to the checkpoints of the specified phase
            contained in the folder
    """
    phase = str(phase)
    all_checkpts = get_checkpoints(folder, checkpt_exts)
    checkpts = []
    for checkpt in all_checkpts:
        try:
            if checkpt.split("phase")[-1].split("_")[0] == phase:
                checkpts.append(checkpt)
        except: pass
    return checkpts

def foldersort(x):
    """
    A sorting key function to order folder names with the format:
    <path_to_folder>/<exp_name>_<exp_num>_<ending_folder_name>/

    x: str
    """
    if x[-1] == "/": x = x[:-1]
    splt = x.split("/")
    if len(splt) > 1: splt = splt[-1].split("_")
    else: splt = splt[0].split("_")
    for i,s in enumerate(splt[1:]):
        try:
            return int(s)
        except:
            pass
    return math.inf

def is_model_folder(path, exp_name=None):
    """
    checks to see if the argued path is a model folder or otherwise.

    path: str
        path to check
    exp_name: str or None
    """
    check_folder = os.path.expanduser(path)
    if exp_name is not None:
        exp_splt = exp_name.split("_")
        # Remove ending slash if there is one
        if check_folder[-1]=="/": check_folder = check_folder[:-1]
        folder_splt = check_folder.split("/")
        folder_splt = folder_splt[-1].split("_")
        match = True
        for i in range(len(exp_splt)):
            if i<len(folder_splt) and exp_splt[i] != folder_splt[i]:
                match = False
        if match: return True
    contents = os.listdir(check_folder)
    for content in contents:
        if ".pt" in content or "hyperparams.txt" == content:
            return True
    return False

def get_model_folders(main_folder, incl_full_path=False):
    """
    Returns a list of paths to the model folders contained within the
    argued main_folder

    main_folder - str
        path to main folder
    incl_full_path: bool
        include extension flag. If true, the expanded paths are
        returned. otherwise only the end folder (i.e.  <folder_name>
        instead of main_folder/<folder_name>)

    Returns:
        list of folder names (see incl_full_path for full path vs end
        point)
    """
    folders = []
    main_folder = os.path.expanduser(main_folder)
    if ".pt" in main_folder[-4:]:
        # if model file, return the corresponding folder
        return [ "/".join(main_folder.split("/")[:-1]) ]
    for d, sub_ds, files in os.walk(main_folder):
        for sub_d in sub_ds:
            check_folder = os.path.join(d,sub_d)
            if is_model_folder(check_folder):
                if incl_full_path:
                    folders.append(check_folder)
                else:
                    folders.append(sub_d)
    if is_model_folder(main_folder): folders.append(main_folder)
    folders = list(set(folders))
    return sorted(folders, key=foldersort)

def load_checkpoint(path, use_best=False, phase=None):
    """
    Loads the save_dict into python. If the path is to a model_folder,
    the loaded checkpoint is the BEST checkpt if available, otherwise
    the checkpt of the last epoch

    Args:
        path: str
            path to checkpoint file or model_folder
        use_best: bool
            if true, will load the best checkpt based on validation metrics
        phase: int or str or None
            the phase of the training that you would like to focus the
            search on. if None, the latest phase is used
    Returns:
        checkpt: dict
            a dict that contains all the valuable information for the
            training.
    """
    path = os.path.expanduser(path)
    hyps = None
    if os.path.isdir(path):
        hyps = cuu.load_json(os.path.join(path, "hyperparams.json"))
        best_path = os.path.join(path,BEST_CHECKPT_NAME)
        if use_best and os.path.exists(best_path):
            path = best_path 
        else:
            if phase is None:
                checkpts = get_checkpoints(path)
            else:
                checkpts = get_phase_checkpoints(path, phase=phase)
            if len(checkpts)==0: return None
            path = checkpts[-1]
    data = torch.load(path, map_location=torch.device("cpu"))
    data["loaded_path"] = path
    if "hyps" not in data: data["hyps"] = hyps
    if "epoch" not in data:
        # Untested!!
        ext = path.split(".")[-1]
        data["epoch"] = int(path.split("."+ext)[0].split("_")[-1])
        torch.save(data, path) 
    return data

def load_model(path, models=None, load_sd=True, use_best=False,
                                           phase=None,
                                           verbose=True):
    """
    Loads the model architecture and state dict from a .pt or .pth
    file. Or from a training save folder. Defaults to the last check
    point file saved in the save folder.

    path: str or dict
        either .pt,.p, or .pth checkpoint file; or path to save folder
        that contains multiple checkpoints. if dict, must be a checkpt
        dict.
    models: dict (just pass `globals()` as the arg) or None
        A dict of the potential model classes. This function is
        easiest if you import each of the model classes in the calling
        script and simply pass `globals()` as the argument for this
        parameter. If None is argued, `globals()` is used instead.

        keys: str
            the class names of the potential models
        vals: Class
            the potential model classes
    load_sd: bool
        if true, the saved state dict is loaded. Otherwise only the
        model architecture is loaded with a random initialization.
    phase: str or int or None
        the desired phase of the experiment. if None, defaults to latest
        phase
    use_best: bool
        if true, will load the best model based on validation metrics
    """
    if type(path) == type(str()):
        path = os.path.expanduser(path)
        hyps = None
        data = load_checkpoint(path,use_best=use_best, phase=phase)
    else: data = path
    if 'hyps' in data:
        kwargs = data['hyps']
    else:
        kwargs = cuu.load_json(os.path.join(path, "hyps.json"))
    if models is None: models = cogmtc.models.__dict__
    model = models[kwargs['model_type']](**kwargs)
    if "state_dict" in data and load_sd:
        print("loading state dict")
        try:
            model.load_state_dict(data["state_dict"])
        except:
            print("failed to load state dict, attempting fix")
            try:
                n,h = data["state_dict"]["cdtnl_lstm.embs.weight"].shape
                model.cdtnl_lstm.embs = torch.nn.Embedding(n,h)
                model.cnn.cdtnl_lstm.embs = torch.nn.Embedding(n,h)
                model.load_state_dict(data["state_dict"])
            except:
                del model.cnn
                model.load_state_dict(data["state_dict"])
            print("succeeded!")
            #sd = data["state_dict"]
            #keys = {*sd.keys(), *model.state_dict().keys()}
            #for k in keys:
            #    if k not in sd: sd[k] = getattr(model, k)
            #    if k not in model: setattr(model, k, sd[k])
    else:
        print("state dict not loaded!")
    return model

def get_hyps(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    folder = os.path.expanduser(folder)
    hyps_json = os.path.join(folder, "hyperparams.json")
    hyps = cuu.load_json(hyps_json)
    return hyps

def load_hyps(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    return get_hyps(folder)

def get_next_exp_num(exp_name):
    """
    Finds the next open experiment id number.

    exp_name: str
        path to the main experiment folder that contains the model
        folders
    """
    folders = get_model_folders(exp_name)
    exp_nums = set()
    for folder in folders:
        exp_num = foldersort(folder)
        exp_nums.add(exp_num)
    for i in range(len(exp_nums)):
        if i not in exp_nums:
            return i
    return len(exp_nums)

def exp_num_exists(exp_num, exp_name):
    """
    Determines if the argued experiment number already exists for the
    argued experiment name.

    exp_num: int
        the number to be determined if preexisting
    exp_name: str
        path to the main experiment folder that contains the model
        folders
    """
    folders = get_model_folders(exp_name)
    for folder in folders:
        num = foldersort(folder)
        if exp_num == num:
            return True
    return False

def make_save_folder(hyps):
    """
    Creates the save name for the model.

    hyps: dict
        keys:
            exp_name: str
            exp_num: int
            search_keys: str
    """
    save_folder = "{}/{}_{}".format(hyps['exp_name'],
                                    hyps['exp_name'],
                                    hyps['exp_num'])
    save_folder += hyps['search_keys']
    return save_folder

