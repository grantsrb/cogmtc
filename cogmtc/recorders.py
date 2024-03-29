from collections import defaultdict
import os
import time
from datetime import datetime
import torch
import numpy as np
import pandas as pd

from cogmtc.utils.save_io import save_checkpt, save_best_checkpt
from cogmtc.utils.utils import try_key
from cogmtc.utils.training import get_exp_num, record_session, get_save_folder, get_git_revision_hash

class Recorder:
    """
    This class assists in recording the settings and training details
    of the experiment.
    """
    def __init__(self, hyps, model, verbose=False):
        """
        Creates a folder under the greater experiment name and saves
        the hyperparams to a json and a text file. Also saves the
        model structure to a text file.

        Args:
            hyps: dict
                keys: str
                    exp_name: str
                        the name of the experiment
            model: torch Module (must implement the repl func)
        """
        self.hyps = hyps
        self.df = None # Used to store a dataframe with the fxn to_df
        self.reset_stats()
        self.best_score = -np.inf
        git_hash = get_git_revision_hash()
        hyps["git_hash"] = git_hash
        # Initialize important variables
        hyps['save_root'] = try_key(hyps, 'save_root', "./")
        hyps["main_path"] = os.path.join(
            hyps["save_root"],
            hyps["exp_name"]
        )
        # save_folder will be present if resuming from specific checkpt
        if "resume_folder" not in hyps or hyps["resume_folder"]=="":
            hyps['exp_num'] = get_exp_num(
                hyps['main_path'],
                hyps['exp_name'],
                offset=try_key(hyps, "exp_num_offset", 0)
            )
            hyps['save_folder'] = get_save_folder(hyps)
        # Create save folder if one doesn't exist
        if not os.path.exists(hyps['save_folder']):
            os.mkdir(hyps['save_folder'])
        if verbose:
            print("Saving to", hyps['save_folder'])
        # writes important details to hyperparams.txt and
        # hyperparams.json within the model folder
        record_session(hyps, model)
        # The logfile will be used to record a quick text output
        # of the training session
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.write_to_log(dt_string+"\n\n")

    def reset_stats(self):
        """
        Resets the metrics dict. Should be called at the end of every
        epoch.
        """
        self.metrics = defaultdict(lambda: [])
        self.epoch_time = time.time()

    def track_loop(self, metrics):
        """
        Used to tack the statistics of the training loop. Simply
        records the values in the argued metrics dict to an internal
        list. This data can then be used to figure out the epoch
        statistics using analyze_stats.

        Args:
            metrics: dict
                keys: str
                vals: float (NOT TENSORS)
        """
        for key in metrics.keys():
            self.metrics[key].append(metrics[key])

    def accumulate_stats(self):
        """
        Calculates average and stdev of the recorded metrics for both
        training and validation. Then combines the results into
        self.metrics
        """
        stats = {}
        for key in self.metrics.keys():
            if len(self.metrics[key]) > 0:
                stats[key + "_avg"] = np.mean(self.metrics[key])
                stats[key + "_std"] = np.std(self.metrics[key])
                stats[key + "_max"] = np.max(self.metrics[key])
                stats[key + "_min"] = np.min(self.metrics[key])
        return stats

    def save_best_model(self, model, score, epoch, phase):
        """
        Saves a checkpoint file for the argued model if the argued
        score is the best yet. Only use for phases 1 and 2.

        Args:
            model: torch Module
            score: float
                validation accuracy or some other score where greater
                is better.
            epoch: int
            phase: int [0,1,2]
                the phase of the training (is it (0) training the
                language network or (1) training the action network or
                (2) both together)
        """
        if score < self.best_score: return
        save_dict = {
            "epoch": epoch,
            "phase": phase,
            "hyps": self.hyps,
            "state_dict": model.state_dict(),
            "optim_dict": None,
            "current_lr": None,
        }
        save_best_checkpt(save_dict, self.hyps["save_folder"])

    def save_epoch_stats(self, phase, epoch, model, optim, verbose=True):
        """
        Saves a checkpoint file to the model folder. The checkpoint
        file includes the statistics of the epoch as well as a state
        dict of both the model and the optimizer.

        Args:
            phase: int [0,1,2]
                the phase of the training (is it (0) training the
                language network or (1) training the action network or
                (2) both together)
            epoch: int
                the epoch count
            model: torch Module
                the trained model. this function will save its
                state_dict
            optim: torch Optimizer
                the training optimizer. this function will save its
                state_dict
        """
        save_name = "checkpt_phase"+str(phase)
        save_name = os.path.join(self.hyps['save_folder'], save_name)
        save_dict = {
            "epoch": epoch,
            "phase": phase,
            "hyps": self.hyps,
            "stats": self.accumulate_stats(),
            "state_dict": model.state_dict(),
            "optim_dict": optim.state_dict(),
            "current_lr": optim.param_groups[0]["lr"],
        }
        save_dict["stats"]["phase"] = phase

        save_checkpt(
            save_dict,
            save_name,
            epoch,
            ext=".pt",
            del_prev_sd=self.hyps['del_prev_sd'],
            best=False
        )
        string = self.make_log_string(save_dict["stats"])
        s = "\nEpoch: "+str(epoch) + " - Phase: " + str(phase)
        s = s + " - LR:" + str(save_dict["current_lr"])
        string = s+string
        if verbose: print(string)
        self.write_to_log(string + "\n\n")

    def make_log_string(self, stats_dict):
        """
        Creates an intelligible string of the statisitics from the
        stats_dict. 

        Args:
            stats_dict: dict
                keys: str
                vals: printable objects
        """
        duration = time.time() - self.epoch_time
        s = ""
        # Get root names for formatting
        keys = sorted(list(stats_dict.keys()))
        roots = set()
        for key in keys:
            if key[:-4] not in roots:
                roots.add(key[:-4])
                s += "\n"+key+": "+str(round(stats_dict[key], 5))
            else:
                s += ", "+key+": "+str(round(stats_dict[key], 5))
        s += "\nEpoch duration: " + str(duration) + " secs"
        return s

    def write_to_log(self, string):
        """
        Writes the string to a new line in the log file. The default
        file name is <model_folder>/log.txt. If verbose, also prints
        the string.
        """
        log_file = os.path.join(
            self.hyps['save_folder'],
            "training_log.txt"
        )
        with open(log_file,'a') as f:
            f.write(str(string)+'\n')

    def to_df(self, **kwargs):
        """
        This function adds the keyword arguments to a running dataframe.
        Each key in the kwargs is used as a column. If the column
        does not exist in the df, it is added with NA as the default
        filler value for all previous entries. Any existing columns
        that are not included in the kwargs keys also default to NA

        Args:
            keyword: list or ndarray
                all lists must be the same length
        """
        new_df = {key: list(kwargs[key]) for key in kwargs.keys()}
        if self.df is None:
            self.df = pd.DataFrame(new_df)
        else:
            self.df = self.df.append(pd.DataFrame(new_df), sort=True)
        df_file = os.path.join(
            self.hyps["save_folder"],
            "validation_stats.csv"
        )
        self.df.to_csv(df_file, sep=",", mode="w")
