import time
import pandas as pd
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from cogmtc.models import NullModel
from cogmtc.envs import SequentialEnvironment
from cogmtc.oracles import *
from cogmtc.utils.utils import try_key, sample_action, zipfian, get_lang_labels, get_loss_and_accs, convert_numeral_array_to_numbers
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import math

INEQUALITY = 0
ENGLISH = 1
PIRAHA = 2
RANDOM = 3
DUPLICATES = 4
NUMERAL = 5

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def get_oracle(env_type, *args, **kwargs):
    """
    Handles getting the appropriate oracle based on the env_type.

    Args:
        env_type: str
            the name of the environment
    Returns:
        oracle: Oracle
            the appropriate oracle for the env_type. oracles take in
            an env and output an action that can be fed directly into
            the env.
    """
    if "gordongame" in env_type:
        return DiscreteOracle(env_type, *args, **kwargs)
    elif "gordoncont" in env_type:
        return ContinuousOracle(env_type, *args, **kwargs)
    else:
      raise NotImplemented(
        "An oracle for {} has not yet been implemented".format(env_type)
      )

def next_state(env, obs_deque, obs, reset, n_targs=None):
    """
    Get the next state of the environment.

    env - environment of interest
    obs_deq - deque of the past n observations
    obs - ndarray returned from the most recent step of the environment
    reset - boolean denoting the reset signal from the most recent step 
            of the environment
    n_targs: int or None
        if int, decides the number of targets for the next episode
    """

    if reset or obs is None:
        obs = env.reset(n_targs=n_targs)
        for i in range(obs_deque.maxlen-1):
            obs_deque.append(np.zeros(obs.shape))
    obs_deque.append(obs)
    state = np.concatenate(obs_deque, axis=0)
    return state

class ExperienceReplay(torch.utils.data.Dataset):
    """
    This class holds the game experience. It holds a number of shared
    tensors. One for each of the rewards, actions, observations, done
    signals, etc. For each shared tensor the experience is held in a
    row corresponding to each parallelized environment. One
    environment exists for each row in the shared tensors. See the
    DataCollector class for more details on how the data collection is
    parallelized.
    """
    def __init__(self,
        hyps,
        share_tensors=True,
        *args,
        **kwargs
    ):
        """
        Args:
            hyps: dict
                exp_len: int
                    the maximum length of the experience tensors
                batch_size: int
                    the number of parallel environments
                inpt_shape: tuple (C, H, W)
                    the shape of the observations. channels first,
                    then height and width
                seq_len: int
                    the length of returned sequences
                randomize_order: bool
                    a bool to determine if the data should be
                    randomized in the iter. if true, data returned
                    from this class's iterable will be in a
                    randomized order.
                share_tensors: bool
                    if true, each tensor within shared_exp is moved to the
                    shared memory for parallel processing
                env_types: str or None
                    used to determine if n_targs, n_items, and n_aligned
                    should be included in the shared_exp dict
                continuous_env: bool
                    a boolean determining if the envs are continuous
                actn_size: int
                    the size of the action space. only necessary if
                    continuous_env is true.
            share_tensors: bool
                set to true if using the experience replay accross
                multiple procs
        Members:
            shared_exp: dict
                keys: str
                    rews: torch float tensor (N, L)
                        the rewards collected by the environments
                    obs: torch float tensor (N, L, C, H, W)
                        the observations collected by the environments
                    dones: torch long tensor (N, L)
                        the done signals collected by the environments
                    actns: torch longtensor (N,L) or floattensor (N,L,A)
                        the actions taken during the data collection.
                        longtensor if discrete env, floattensor if
                        continuous
                    n_targs: torch long tensor (N, L) or None
                        the number of goal targets if using gordongames
                        environment
                    n_items: torch long tensor (N, L) or None
                        the number of items in the env if using
                        gordongames environment
                    n_aligned: torch long tensor (N, L) or None
                        the number of aligned items in the env if using
                        gordongames environment
        """
        self.hyps = hyps
        self.exp_len = self.hyps["exp_len"]
        self.batch_size = self.hyps["batch_size"]
        self.inpt_shape = self.hyps["inpt_shape"]
        self.seq_len = self.hyps["seq_len"]
        self.randomize_order = self.hyps["randomize_order"]

        # Hold outs are used in the get_drops function as a way
        # to determine if a word should ever be trained on. 
        # The hold_lang default to hold_outs if nothing is argued.
        # The setting of hold_lang is handled in the
        # `hyps_error_catching` fxn in `training.py`
        self.hold_lang = hyps["hold_lang"]
        if hyps["hold_lang"] is not None:
            self.hold_lang = torch.LongTensor(list(hyps["hold_lang"]))
        print("Hold lang:", self.hold_lang)

        self.roll_data = try_key(self.hyps, "roll_data", True)
        self.share_tensors = share_tensors
        assert self.exp_len > self.seq_len,\
            "sequence length must be less than total experience length"
        self.shared_exp = {
            "obs": torch.zeros((
                    self.batch_size,
                    self.exp_len,
                    *self.inpt_shape
                )).float(),
            "rews": torch.zeros((
                    self.batch_size,
                    self.exp_len
                )).float(),
            "dones": torch.zeros((
                    self.batch_size,
                    self.exp_len
                )).long(),
            "actns": torch.zeros((
                    self.batch_size,
                    self.exp_len
                )).long(),
        }
        if self.hyps["continuous_env"]:
            self.shared_exp["actns"] = torch.zeros((
                        self.batch_size,
                        self.exp_len,
                        self.hyps["actn_size"]
                    )).float()
        self.info_keys = [
            "n_targs","n_items","n_aligned","grabs","disp_targs",
            "is_animating",
        ]
        for key in self.info_keys:
            self.shared_exp[key] = torch.zeros((
                self.batch_size,
                self.exp_len
            )).long()
        if self.share_tensors:
            for key in self.shared_exp.keys():
                self.shared_exp[key].share_memory_()
        self.harvest_exp()

    def harvest_exp(self, phase=None):
        """
        Copys the shared tensors so that the runners can continue
        collecting without changing the data.

        Args:
            phase: int or None
                the phase of the training. only necessary if blind_lang
                is true
        Returns:
            exp: dict of tensors
                deep copies the shared experience
        """
        self.exp = {
          k: v.detach().data.clone() for k,v in self.shared_exp.items()
        }
        if self.hyps["blind_lang"] and phase == 0:
            count_list = torch.arange(self.hyps["lang_range"][1])
            l = self.exp["n_items"].shape[-1]
            n = l//len(count_list)+1
            b = len(self.exp["n_items"])
            count_list = count_list[None].repeat((b,n))
            self.exp["n_items"][:] = count_list[:,:l]
            #zeros = torch.zeros_like(count_list)
            #for i in range(b):
            #    if i > 0: self.exp["n_items"][i,:i] = zeros[:i]
            #    self.exp["n_items"][i,i:] = count_list[:l-i]

        self.exp["lang_labels"] = get_lang_labels(
            self.exp["n_items"],
            self.exp["n_targs"],
            max_label=self.hyps["lang_size"]-1,
            use_count_words=self.hyps["use_count_words"],
            max_char_seq=self.hyps["max_char_seq"]
        )
        return self.exp

    def __len__(self):
        raw_len = len(self.shared_exp["rews"][0]) - self.seq_len + 1
        #raw_len = len(self.shared_exp["rews"][0])
        if self.roll_data:
            return raw_len
        return int(raw_len//self.seq_len)

    def __getitem__(self, idx):
        """
        Returns a chunk of data with the sequence length including all
        environments.

        Args:
            idx: int
                this value is multiplied by the sequence length if
                roll_data is false
        Returns:
            data: dict
                keys: str
                    obs: torch float tensor (N, S, C, H, W)
                    rews: torch float tensor (N, S)
                    dones: torch long tensor (N, S)
                    actns: torch long tensor (N, S)
        """
        if not self.roll_data:
            idx = idx*self.seq_len
        data = dict()
        for key in self.exp.keys():
            data[key] = self.exp[key][:, idx: idx+self.seq_len]
        data["drops"] = self.get_drops(
            self.hyps,
            data["grabs"],
            data["is_animating"]
        )
        if self.hold_lang is not None:
          data["drops"][torch.isin(data["n_items"],self.hold_lang)] = 0
        return data

    def __iter__(self):
        """
        Uses a permutation to track which index is next.

        Note that if __iter__ is called a second time, then any
        exiting iterable of this class will also be affected!
        """
        if self.randomize_order:
            self.idx_order = torch.randperm(self.__len__()).long()
        else:
            self.idx_order = torch.arange(self.__len__()).long()
        self.idx = 0
        return self

    def __next__(self):
        """
        Returns:
            data: dict
                keys: str
                    obs: torch float tensor (N, S, C, H, W)
                    rews: torch float tensor (N, S)
                    dones: torch long tensor (N, S)
                    actns: torch long tensor (N, S)
        """
        if not hasattr(self, "idx_order"):
            self.__iter__()
        if self.idx < self.__len__():
            idx = self.idx_order[self.idx]
            self.idx += 1
            return self.__getitem__(idx)
        else:
            raise StopIteration

    @staticmethod
    def get_drops(hyps, grabs, is_animating):
        """
        Returns a tensor denoting steps in which the agent dropped an
        item. This always means that the player is still on the item
        when the prediction happens.
        
        WARNING: the returned tensor has been bastardized to simply
        denote when the agent should make a language prediction about
        n_items.

        The tensor returned is also actually a LongTensor, not a Bool
        Tensor. Assumes 1 means PILE, and 2 means BUTTON and 3 means
        ITEM within the grabs tensor.

        For variants 4 and 7, drops includes any frame in which the
        targets are displayed.

        Args:
            hyps: dict
                the hyperparameters
                langall: bool
                count_targs: bool
                drops_perc_threshold: float
                lang_targs_only: int
                    if 0, does nothing. If 1, will only return drops
                    where is_animating is true. This argument is
                    overridden by langall being true.
                    count_targs is overridden by this argument.
            grabs: Long Tensor (B,N)
                a tensor denoting the item grabbed by the agent at
                each timestep. Assumes 1 means PILE, and 2 means BUTTON
                and 3 means ITEM
            is_animating: torch LongTensor (..., N)
                0s denote the environment was not displaying the targets
                anymore. 1s denote the targets were displayed
        Returns:
            drops: Long Tensor (B,N)
                a tensor denoting if the agent dropped an item with a 1,
                0 otherwise. See WARNING in description
        """
        if type(grabs) == type(np.asarray([])):
            grabs = torch.from_numpy(grabs).long()
        if try_key(hyps, "langall", False):
            drops = torch.ones_like(grabs)
        elif try_key(hyps, "lang_targs_only", 0) == 1:
            drops = is_animating.clone()
        else:
            block = len(grabs)//len(hyps["env_types"])
            drops = torch.zeros_like(grabs).long()
            for i,env_type in enumerate(hyps["env_types"]):
                temp = {**hyps, "env_type": env_type}
                fxn = ExperienceReplay.get_drops_helper
                drops[i*block:(i+1)*block] = fxn(
                    temp,
                    grabs[i*block:(i+1)*block],
                    is_animating[i*block:(i+1)*block]
                )
        return drops

    @staticmethod
    def get_drops_helper(hyps, grabs, is_animating):
        """
        Assists the get_drops function.

        Args:
            hyps: dict
                the hyperparameters
                langall: bool
                count_targs: bool
                drops_perc_threshold: float
                lang_targs_only: int
                    if 0, does nothing. If 1, will only return drops
                    where is_animating is true. This argument is
                    overridden by langall being true.
                    count_targs is overridden by this argument.
            grabs: Long Tensor (B,N)
                a tensor denoting the item grabbed by the agent at
                each timestep. Assumes 1 means PILE, and 2 means BUTTON
                and 3 means ITEM
            is_animating: torch LongTensor (..., N)
                0s denote the environment was not displaying the targets
                anymore. 1s denote the targets were displayed
        Returns:
            drops: Long Tensor (B,N)
                a tensor denoting if the agent dropped an item with a 1,
                0 otherwise. See WARNING in description
        """
        drops = grabs.clone().long()
        if hyps["env_type"] in {"gordongames-v4", "gordongames-v8"}:
            drops[drops>0] = 1
            if try_key(hyps, "count_targs", True):
                drops = drops | (is_animating>0)
            return drops
        drops[grabs!=3] = 0
        drops[grabs==3] = 1
        # Looks for situations where the sum drops off to 0
        if len(grabs.shape)==2:
            drops[:, 1:] = drops[:, :-1] - drops[:, 1:]
            drops[:,0] = 0
            drops = torch.roll(drops, 1, -1)
            drops[:,0] = 0
        elif len(grabs.shape)==1:
            drops[1:] = drops[:-1] - drops[1:]
            drops[0] = 0
            drops = torch.roll(drops, 1, -1)
            drops[0] = 0
        drops[drops!=1] = 0
        drops[drops>0] = 1
        if try_key(hyps, "count_targs", True):
            drops = drops | (is_animating>0)
        # In case less than 5% of the batch are drops, we set the last
        # column to 1
        perc_threshold = try_key(hyps,"drops_perc_threshold",0)
        if drops.sum()<=(perc_threshold*drops.numel()):
            if perc_threshold == 0: perc_threshold = 0.1
            rand = torch.rand_like(drops.float())
            rand[rand<=perc_threshold] = 1
            rand[rand!=1] = 0
            drops = drops | rand.long()
        return drops

class DataCollector:
    """
    This class collects the training data by rolling out multiple
    environments in parallel. It places the data into the shared
    tensors within the experience replay.

    The data collector spawns multiple runners who collect data from
    their respective environments.
    """
    def __init__(self, hyps):
        """
        Creates the runners and initiates the initial data collection.
        Separate from the __init__ function because it's most efficient
        to get the observation size for the experience replay from
        the validation runner which is created internally

        Args:
            hyps: dict
                keys: str
                    batch_size: int
                        the number of parallel environments
                    env_types: int
                        the number of runners to instantiate
        """
        self.n_envs = len(hyps["env_types"])
        hyps["batch_size"]=(hyps["batch_size"]//self.n_envs)*self.n_envs
        self.hyps = hyps
        self.batch_size = self.hyps['batch_size']
        # Create gating mechanisms
        self.gate_q = mp.Queue(self.n_envs)
        self.stop_q = mp.Queue(self.n_envs)
        self.val_gate_q = mp.Queue(1)
        self.val_stop_q = mp.Queue(1)
        # Used to update training phase information accross procs
        self.phase_q = mp.Queue(1)
        self.phase_q.put(try_key(hyps, "first_phase", 0))
        self.terminate_q = mp.Queue(1)
        self.terminate_q.put(0)
        # Used to communicate results of validation training to
        # main proc
        self.val_q = mp.Queue(1)
        # Get observation, actn, and lang shapes
        self.validator = ValidationRunner(
            self.hyps,
            gate_q=self.val_gate_q,
            stop_q=self.val_stop_q,
            phase_q=self.phase_q,
            terminate_q=self.terminate_q,
            val_q=self.val_q
        )
        print("val runner created")
        self.validator.create_new_env()
        self.obs_shape = self.validator.env.shape
        self.hyps['inpt_shape'] = self.validator.state_bookmark.shape
        self.hyps["actn_size"] = self.validator.env.actn_size
        self.hyps["continuous_env"] = self.validator.env.is_continuous
        if self.hyps["continuous_env"]: self.hyps["loss_fxn"]="mse_loss"
        else: self.hyps["loss_fxn"] = "cross_entropy"
        self.validator.loss_fxn = getattr(F, self.hyps["loss_fxn"])
        self.hyps["lang_range"] = try_key(
            self.hyps,
            "lang_range",
            self.hyps["targ_range"]
        )
        if self.hyps["lang_range"] is None:
            self.hyps["lang_range"] = self.hyps["targ_range"]
        lang_range = self.hyps["lang_range"]
        self.hyps["lang_size"] = lang_range[1]+1 # plus one includes zero
        self.hyps["max_char_seq"] = 1
        # If comparison or piraha language, must change lang_size
        if int(self.hyps["use_count_words"]) == INEQUALITY:
            self.hyps["lang_size"] = 3
        elif int(self.hyps["use_count_words"]) == PIRAHA:
            self.hyps["lang_size"] = 4
        elif int(self.hyps["use_count_words"]) == DUPLICATES:
            self.hyps["lang_size"] = self.hyps["lang_size"]*2
        if int(self.hyps["use_count_words"]) == NUMERAL:
            self.hyps["numeral_base"]=try_key(self.hyps,"numeral_base",4)
            base = self.hyps["numeral_base"]
            # Add 1 to base for the STOP token
            self.hyps["lang_size"] = base+1
            # Attempting to determine the maximum character sequence
            # length for predicting the systematic number system with
            # base `hyps["numeral_base"]`. We add 1 to the sequence
            # length for the STOP token and 1 more for testing
            # generalization
            mcs = int(math.ceil(math.log((lang_range[1]+1),base))+2)
            self.hyps["max_char_seq"] = mcs
        else:
            self.hyps["numeral_base"] = None
            self.hyps["max_char_seq"] = None
            self.hyps["lstm_lang"] = None
        self.validator.hyps["max_char_seq"] = self.hyps["max_char_seq"]
        self.validator.hyps["numeral_base"] = self.hyps["numeral_base"]
        # Initialize Experience Replay
        self.exp_replay = ExperienceReplay(hyps)
        print("exp_replay created")
        # Initialize runners
        # We add one here because the validation environment defaults
        # to the argued seed without offset
        self.env_types = self.hyps["env_types"]
        self.runners = []
        offset = try_key(self.hyps, 'runner_seed_offset', 0)
        for i,env_type in enumerate(self.env_types):
            seed = self.hyps["seed"] + offset + i + 1
            temp_hyps = {**self.hyps,"seed":seed,"env_type":env_type}
            runner = Runner(
                i,
                self.exp_replay.shared_exp,
                temp_hyps,
                self.gate_q,
                self.stop_q,
                self.phase_q,
                terminate_q=self.terminate_q
            )
            self.runners.append(runner)
        print("runners created")

        # Initiate Data Collection
    def init_runner_procs(self, model):
        """
        Initializes the runner processes

        Args:
            model: torch Module
                make sure the model's weights are shared so that they
                can be updated over the course of the training.
        """
        if not hasattr(self, "procs"):
            self.procs = []
        for i in range(len(self.runners)):
            proc = mp.Process(
                target=self.runners[i].run,
                args=(model,)
            )
            proc.start()
            self.procs.append(proc)

    def init_validator_proc(self, model):
        """
        Initializes the validation runner process

        Args:
            model: torch Module
                make sure the model's weights are shared so that they
                can be updated over the course of the training.
        """
        if not hasattr(self, "procs"):
            self.procs = []
        proc = mp.Process(
            target=self.validator.run,
            args=(model,)
        )
        proc.start()
        self.procs.append(proc)

    def await_runners(self):
        for i in range(self.n_envs):
            self.stop_q.get()

    def dispatch_runners(self):
        for i in range(self.n_envs):
            self.gate_q.put(i)

    def await_validator(self):
        self.val_stop_q.get()

    def dispatch_validator(self, epoch):
        self.val_gate_q.put(epoch)

    def terminate_procs(self):
        """
        Includes the validation runner process
        """
        self.terminate_q.get()
        self.terminate_q.put(1)
        self.dispatch_runners()
        self.dispatch_validator(np.inf)
        for proc in self.procs:
            proc.join()

    def update_phase(self, phase):
        old_phase = self.phase_q.get()
        self.phase_q.put(phase)

def sample_zipfian(hyps):
    """
    A helper function to sample from the zipfian distribution according
    to the hyperparameters.

    Args:
        hyps: dict
            the hyperparameters
    Returns:
        sample: int or None
            the sampled value if zipf_order is not none
    """
    order = try_key(hyps, "zipf_order", None)
    if order is not None and order > 0:
        low, high = hyps["targ_range"]
        return zipfian(low, high, order)
    return None

class Runner:
    def __init__(self, idx,
                       shared_exp,
                       hyps,
                       gate_q,
                       stop_q,
                       phase_q,
                       terminate_q):
        """
        Args:
            idx: int
                an identifier for the runner
            hyps: dict
                keys: str
                    "gamma": reward decay coeficient
                    "exp_len": number of steps to be taken in the
                                environment
                    "n_frame_stack": number of frames to stack for
                                     creation of the mdp state
                    "preprocessor": function to preprocess raw
                                    observations
                    "env_type": type of gym environment to be interacted
                                with. Follows OpenAI's gym api.
                    "hold_outs": set of ints
                        a set of target values that should not be
                        included in the training data
                    "hold_actns": bool
                        if false, hold_outs is set to None
            shared_exp: dict
                keys: str
                vals: shared torch tensors
                    "obs": Collects the MDP states at each timestep t
                    "rews": Collects float rewards collected at each
                            timestep t
                    "dones": Collects the dones collected at each
                             timestep t
                    "actns": Collects actions performed at each
                             timestep t
                    "n_targs": Collects the number of targets for the
                               episode if using gordongames variant
                    "n_items": Collects the number of items over the
                               course of the episode if using
                               gordongames variant
                    "n_aligned": Collects the number of items aligned
                               with targets over the course of the
                               episode if using gordongames variant
                    "grabs": Collects information into whether the agent
                            is grabbing or not.
            gate_q: multiprocessing Queue.
                Allows main process to control when rollouts should be
                collected.
            stop_q: multiprocessing Queue.
                Used to indicate to main process that a rollout has
                been collected.
            phase_q: multiprocessing Queue.
                Used to indicate from the main process that the phase
                has changed.
            terminate_q: multiprocessing Queue.
                Used to indicate the end of the training from the main
                process.
        """
        self.hyps = {**hyps}
        self.hyps["hold_outs"] = self.hyps["hold_actns"]
        self.shared_exp = shared_exp
        self.idx = idx
        self.gate_q = gate_q
        self.stop_q = stop_q
        self.phase_q = phase_q
        self.terminate_q = terminate_q
        self.obs_deque = deque(maxlen=hyps['n_frame_stack'])
        env_type = self.hyps['env_type']
        self.oracle = get_oracle(**self.hyps)
        self.rand = np.random.default_rng(self.hyps["seed"])

    def create_new_env(self, n_targs=None, env_type=None):
        """
        This function simplifies making a new environment and storing
        all the variables associated with it. It uses the language
        target range or the action target range depending on the phase
        of the experiment. Phase 0 means language, anything else means
        action.

        Args:
            n_targs: int
            env_type None or str
                if None, defaults to whatever env_type is in the
                hyperparams. Otherwise the argued env_type is used.
        """
        # Set defaults
        if try_key(self.hyps, "actn_range", None) is None:
            self.hyps["actn_range"] = self.hyps["targ_range"]
        if try_key(self.hyps, "lang_range", None) is None:
            self.hyps["lang_range"] = self.hyps["targ_range"]

        if self.phase == 0:
            # Set environment targ range to language range
            self.hyps["targ_range"] = self.hyps["lang_range"]
        else:
            # Set environment targ range to action range
            self.hyps["targ_range"] = self.hyps["actn_range"]
        self.hyps["seed"] += int(self.rand.random()*100000)
        hyps = {**self.hyps}
        if n_targs is not None:
            hyps["targ_range"] =  (n_targs, n_targs)
        if env_type is not None:
            hyps["env_type"] = env_type
        self.env = SequentialEnvironment(**hyps)
        state = next_state(
            self.env,
            self.obs_deque,
            obs=None,
            reset=True,
            n_targs=sample_zipfian(hyps)
        )
        self.state_bookmark = state
        self.h_bookmark = None
        return state

    def set_random_seed(self, seed):
        self.rand = np.random.default_rng(seed)

    def run(self, model=None):
        """
        run is the entry function to begin collecting rollouts from the
        environment. gate_q indicates when to begin collecting a
        rollout and is controlled from the main process. The stop_q is
        used to indicate to the main process that a new rollout has
        been collected.
        """
        self.set_random_seed(self.hyps["seed"])
        self.phase = try_key(self.hyps, "first_phase", 0)
        self.model = model
        if model is None:
            self.model = NullModel(**self.hyps)
        bsize = self.hyps["batch_size"]
        n_envs = len(self.hyps["env_types"])
        state = self.create_new_env()
        self.ep_rew = 0
        while True:
            with torch.no_grad():
                # Await collection signal from main proc
                idx = self.gate_q.get()
                if idx != self.idx:
                    self.gate_q.put(idx)
                else:
                    terminate = self.terminate_q.get()
                    self.terminate_q.put(terminate)
                    if terminate==1:
                        if hasattr(self, "shared_exp"):
                            keys = list(self.shared_exp.keys())
                            for k in keys:
                                v = self.shared_exp[k]
                                del v
                                del self.shared_exp[k]
                            del self.shared_exp
                        if hasattr(self, "model"):
                            del self.model
                        print("Terminating runner", self.idx)
                        del self.gate_q
                        del self.stop_q
                        del self.phase_q
                        del self.terminate_q
                        return
                    # Change phase if necessary
                    phase = self.phase_q.get()
                    self.phase_q.put(phase)
                    if self.phase != phase:
                        self.phase = phase
                        state = self.create_new_env()
                    # Collect rollouts
                    for i in range(bsize//n_envs):
                        idx = self.idx*(bsize//n_envs) + i
                        if idx < bsize:
                            self.rollout(idx, self.model)
                    # Signals to main process that data has been collected
                    self.stop_q.put(self.idx)

    def rollout(self, idx, model):
        """
        rollout handles the actual rollout of the environment. It runs
        for n steps in the game. Collected data is placed into the
        shared_exp dict in the row corresponding to the argued idx.

        Args:
            idx: int
                identification number distinguishing the row of the
                shared_exp designated for this rollout
        """
        model.eval()
        debug = try_key(self.hyps,"debug",False) and model.trn_whls<1
        if try_key(self.hyps, "reset_trn_env", False):
            state = next_state(
                self.env,
                self.obs_deque,
                obs=None,
                reset=True,
                n_targs=sample_zipfian(self.hyps)
            )
            model.reset(1)
        else:
            state = self.state_bookmark
            self.handle_model_bookmark(model)
        exp_len = self.hyps['exp_len']
        with torch.no_grad():
            idxs = model.cdtnl_idxs[self.idx][None]
            cdtnl = model.cdtnl_lstm(idxs)
        for i in range(exp_len):
            # Collect the state of the environment
            t_state = torch.FloatTensor(state) # (C, H, W)
            self.shared_exp["obs"][idx,i] = t_state
            # Get actn
            actn_targ = self.oracle(self.env, state=t_state) # int
            if self.rand.random()>model.trn_whls:
                inpt = t_state[None].to(DEVICE)
                actn_pred, _ = model.step(inpt, cdtnl)
                actn = self.get_action(actn_pred)
            else:
                actn_pred = None
                actn = actn_targ
            # Step the environment
            obs, rew, done, info = self.env.step(actn)
            if self.env.is_continuous:
                actn_targ = torch.FloatTensor(actn_targ)
            # Collect data
            self.shared_exp['rews'][idx,i] = rew
            self.shared_exp['dones'][idx,i] = float(done)
            self.shared_exp['actns'][idx,i] = actn_targ
            self.shared_exp["n_items"][idx,i] = info["n_items"]
            self.shared_exp["n_targs"][idx,i] = info["n_targs"]
            self.shared_exp["n_aligned"][idx,i] = info["n_aligned"]
            self.shared_exp["grabs"][idx,i] = int(info["grab"])
            self.shared_exp["disp_targs"][idx,i] = int(info["disp_targs"])
            self.shared_exp["is_animating"][idx,i] = int(info["is_animating"])

            state = next_state(
                self.env,
                self.obs_deque,
                obs=obs,
                reset=done,
                n_targs=sample_zipfian(self.hyps)
            )
            if done: model.reset(1)
            if debug and idx==0:
                #env.render()
                print("Actn:", actn)
                print("Actn Targ:", actn_targ)
                print("Model:", actn_pred)
                plt.imshow(obs.squeeze())
                plt.show()

        self.state_bookmark = state
        self.handle_model_bookmark(model)

    def handle_model_bookmark(self, model):
        """
        This is a helper function to handle bookmarking of the model's
        recurrent memory vectors. It first records the current model
        recurrent state, then updates the model with the bookmarked
        state and finally uses the recorded recurrent state to update
        the bookmark.

        Args:
            model: torch.nn.Module
        """
        h_bookmark = None
        if hasattr(model, "hs"):
            h_bookmark = (
                [h.detach().data for h in model.hs],
                [c.detach().data for c in model.cs]
            )
        elif hasattr(model, "h"):
            h_bookmark = (
                model.h.detach().data,
                model.c.detach().data
            )

        if self.h_bookmark is None:
            model.reset(1)
        elif hasattr(model, "hs"):
            model.hs, model.cs = self.h_bookmark
        elif hasattr(model, "h"):
            model.h,model.c = self.h_bookmark

        self.h_bookmark = h_bookmark

    def get_action(self, actn_pred):
        """
        Outputs the appropriate action from the actn_pred to be used
        by the environment.

        Args:
            actn_pred: torch FloatTensor (1,A)
        Returns:
            actn: int or float ndarray (A,)
                the action to be used by the environment
        """
        if self.env.is_continuous:
            return actn_pred.squeeze().cpu().data.numpy()
        if try_key(self.hyps, "val_max_actn", False):
            actn = torch.argmax(actn_pred[0]).item()
        else:
            actn = sample_action(
                F.softmax(actn_pred, dim=-1)
            ).item()
        return actn

class ValidationRunner(Runner):
    def __init__(self, hyps,
                       gate_q=None,
                       stop_q=None,
                       phase_q=None,
                       terminate_q=None,
                       phase=0,
                       val_q=None):
        """
        Args:
            hyps: dict
                keys: str
                    "gamma": reward decay coeficient
                    "exp_len": number of steps to be taken in the
                                environment
                    "n_frame_stack": number of frames to stack for
                                     creation of the mdp state
                    "preprocessor": function to preprocess raw
                                    observations
                    "env_type": type of gym environment to be interacted
                                with. Follows OpenAI's gym api.
            gate_q: multiprocessing Queue.
                Allows main process to control when rollouts should be
                collected.
            stop_q: multiprocessing Queue.
                Used to indicate to main process that a rollout has
                been collected.
            phase_q: multiprocessing Queue.
                Used to indicate from the main process that the phase
                has changed.
            phase: int
                the initial phase
            val_q: multiprocessing Queue.
                Used to indicate to the main process the total accuracy
                acheived by the latest validation.
        """
        self.hyps = {**hyps}
        self.seed = self.hyps["seed"]
        self.gate_q = gate_q
        self.stop_q = stop_q
        self.phase_q = phase_q
        self.terminate_q = terminate_q
        self.val_q = val_q
        # Default to targ_range if no val_targ_range is specified
        if try_key(self.hyps, "val_targ_range", None) is None:
            self.hyps["val_targ_range"] = self.hyps["targ_range"]
        self.hyps["targ_range"] = self.hyps["val_targ_range"]
        self.hyps["actn_range"] = self.hyps["val_targ_range"]
        self.hyps["lang_range"] = self.hyps["val_targ_range"]
        self.hyps["hold_lang"] = None
        self.hyps["hold_actns"] = None
        self.hyps["hold_outs"] = set()
        print("Validation runner target range:",self.hyps["targ_range"])
        self.phase = phase
        self.obs_deque = deque(maxlen=hyps['n_frame_stack'])
        self.env_types = self.hyps["env_types"]
        self.hyps["env_type"] = self.env_types[0]
        self.oracles = {}
        for env_type in self.env_types:
            temp_hyps = {**self.hyps, "env_type": env_type}
            self.oracles[env_type] = get_oracle(**temp_hyps)
        self.rand = np.random.default_rng(self.hyps["seed"])
        self.ep_idx = 0 # Used to track which data goes with which ep
        self.loss_fxn = lambda x: x
        if "loss_fxn" in self.hyps:
            self.loss_fxn = getattr(F, self.hyps["loss_fxn"])

    def run(self, model=None):
        """
        run is the entry function to begin collecting rollouts from the
        environment. gate_q indicates when to begin collecting a
        rollout and is controlled from the main process. The stop_q is
        used to indicate to the main process that a new rollout has
        been collected.
        """
        self.set_random_seed(self.hyps["seed"])
        self.phase = try_key(self.hyps, "first_phase", 0)
        self.model = model
        if model is None:
            self.model = NullModel(**self.hyps)
        state = self.create_new_env()
        self.ep_rew = 0
        while True:
            with torch.no_grad():
                # Await collection signal from main proc
                epoch = self.gate_q.get()
                terminate = self.terminate_q.get()
                self.terminate_q.put(terminate)
                if terminate==1:
                    if hasattr(self, "model"):
                        del self.model
                    print("Terminating validator process")
                    del self.gate_q
                    del self.stop_q
                    del self.phase_q
                    del self.terminate_q
                    return
                # Change phase if necessary
                phase = self.phase_q.get()
                self.phase_q.put(phase)
                if self.phase != phase:
                    self.phase = phase
                # Collect rollouts
                self.rollout(epoch, self.model)
                # Signals to main process that data has been collected
                self.stop_q.put(epoch)

    def rollout(self, epoch, model, *args, **kwargs):
        """
        rollout handles running the environment using the model's
        predictions to direct the game's MDP. This rollout function
        steps through each target quantity and runs the episodes to
        completion. It then saves the language predictions and the
        outcomes of the episodes.

        Args:
            epoch: int
                the current epoch
            model: torch Module
        """
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        model.eval()

        # Reset every on every validation run
        rainj = range(
            self.hyps["targ_range"][0],
            self.hyps["targ_range"][1]+1
        )
        if self.hyps["exp_name"] == "test":
            rainj = range(2,3)
        self.hyps["seed"] = self.seed
        avg_acc = 0
        avg_loss = 0
        for env_type in self.env_types:
            self.oracle = self.oracles[env_type]
            for n_targs in rainj:
                data = self.collect_data(model, env_type, n_targs)
                idx = data["dones"]==1
                acc = (data["n_items"][idx] == n_targs).float().mean()
                avg_acc += acc.item()
                lang_labels = get_lang_labels(
                    data["n_items"],
                    data["n_targs"],
                    max_label=model.lang_size-1,
                    use_count_words=self.hyps["use_count_words"],
                    max_char_seq=self.hyps["max_char_seq"]
                )
                drops = ExperienceReplay.get_drops(
                    self.hyps,
                    data["grabs"],
                    data["is_animating"]
                )
                data["lang_targs"] = lang_labels
                data["drops"] = drops

                print("\nValidated",env_type, "| NTargs:", n_targs)
                # Save epoch results
                self.save_epoch_data(
                    data,
                    epoch,
                    self.phase,
                    n_targs,
                    env_type,
                    base=self.hyps["numeral_base"],
                    max_char_seq=self.hyps["max_char_seq"]
                )

                # Convert language predictions to appropriate form
                avg = data["lang_preds"].mean(0)
                if self.hyps["use_count_words"]==NUMERAL:
                    base = self.hyps["numeral_base"]
                    lang = avg.reshape(len(avg), -1, base+1)
                    lang = torch.argmax(lang, dim=-1) # (N,C)
                    lang = convert_numeral_array_to_numbers(
                        lang.detach().cpu(), base=base
                    )
                    lang_labels = convert_numeral_array_to_numbers(
                        lang_labels, base=base
                    )
                else:
                    lang = torch.argmax(avg, dim=-1) # (N,)
                data["lang_preds"] = lang

                # Save the results
                self.save_lang_data(
                  data, lang_labels, drops, epoch, self.phase, env_type
                )
                self.save_actn_data(data, epoch, self.phase, env_type)
        avg_acc = avg_acc/(len(rainj)*len(self.env_types))
        print("Total Val Acc:", avg_acc)
        if self.val_q is not None:
            if not self.val_q.empty():
                _ = self.val_q.get()
            self.val_q.put(avg_acc)

    def save_lang_data(self,
                       data,
                       labels,
                       drops,
                       epoch,
                       phase,
                       env_type,
                       save_name="validation_lang.csv"):
        """
        Saves the stats at the end of each episode collected from the
        rollouts. Saves the data as a dataframe called `save_name`
        within the model's save_folder.

        Args:
            data: dict
                lang_preds: torch FloatTensor (N,)
                    the max of the language prediction logits.
                n_targs: torch LongTensor (N,)
                    the number of target objects on the grid at this step
                    of the episode
                n_items: torch LongTensor (N,)
                    the number of item objects on the grid at this step
                    of the episode
                n_aligned: torch LongTensor (N,)
                    the number of aligned item objects on the grid at
                    this step of the episode
                dones: torch LongTensor (N,)
                    1 if episode ended on this step, 0 otherwise
                is_animating: torch LongTensor (N,)
                    1 if step is animation step
            labels: tensor (
            epoch: int
            phase: int
            save_name: str
            env_type: str
        """
        inpts = {
            "n_items":None,
            "n_targs":None,
            "n_aligned":None,
            "pred":None,
            "label":None,
            "done":None,
            "is_animating":None,
            "ep_idx": None,
        }
        idxs = (drops>=1)
        if idxs.float().sum() <= 1: return # ensure some data to record

        lang = data["lang_preds"]
        inpts["pred"] = lang[idxs]
        inpts["label"] = labels[idxs]
        inpts["n_targs"] = data["n_targs"][idxs]
        inpts["n_items"] = data["n_items"][idxs]
        inpts["n_aligned"] = data["n_aligned"][idxs]
        inpts["done"] = data["dones"][idxs]
        inpts["is_animating"] = data["is_animating"][idxs]
        inpts["ep_idx"] = data["ep_idx"][idxs]
        inpts = {k:v.cpu().data.numpy() for k,v in inpts.items()}

        df = pd.DataFrame(inpts)
        df["epoch"] = epoch
        df["phase"] = self.phase
        df["env_type"] = env_type
        path = os.path.join(
            self.hyps["save_folder"],
            save_name
        )
        header = not os.path.exists(path)
        df.to_csv(
            path,
            sep=",",
            header=header,
            mode="a"
        )

    def save_actn_data(self,
                       data,
                       epoch,
                       phase,
                       env_type,
                       save_name="validation_stats.csv"):
        """
        Saves the stats at the end of each episode collected from the
        rollouts. Saves the data as a dataframe called `save_name`
        within the model's save_folder.

        Args:
            data: dict
            epoch: int
            phase: int
            env_type: str
            save_name: str
        """
        keys = ["n_items", "n_targs", "n_aligned", "ep_idx"]
        dones = data["dones"].reshape(-1)
        inpts = {
            key: data[key].reshape(-1)[dones==1] for key in keys
        }
        inpts = {k:v.cpu().data.numpy() for k,v in inpts.items()}
        df = pd.DataFrame(inpts)
        df["epoch"] = epoch
        df["phase"] = self.phase
        df["env_type"] = env_type
        path = os.path.join(
            self.hyps["save_folder"],
            save_name
        )
        header = not os.path.exists(path)
        df.to_csv(
            path,
            sep=",",
            header=header,
            mode="a"
        )

    def save_epoch_data(self,
                        data,
                        epoch,
                        phase,
                        n_targs,
                        env_type,
                        base=None,
                        max_char_seq=None,
                        save_name="epoch_stats.csv"):
        """
        Saves the loss and acc stats averaged over all episodes in the
        validation for a given n_target value. Saves the data as a
        dataframe called `save_name` within the model's save_folder.

        Args:
            data: dict
            epoch: int
            phase: int
            n_targs: int
                the number of targets that this data pertains to
            env_type: str
            base: int or None
                the numeral base of the system if use_count_words==
                NUMERAL
            max_char_seq: int or None
                the number of numeral predictions per sample if using
                use_count_words==NUMERAL
            save_name: str
        """
        with torch.no_grad():
            _,losses,accs = get_loss_and_accs(
                phase=phase,
                actn_preds=data["actn_preds"],
                lang_preds=data["lang_preds"],
                actn_targs=data["actn_targs"],
                lang_targs=data["lang_targs"],
                drops=data["drops"],
                n_targs=data["n_targs"],
                n_items=data["n_items"],
                prepender="",
                loss_fxn=self.loss_fxn,
                base=base,
                max_char_seq=max_char_seq
            )
            s = ""
            for k in accs:
                if "_actn_acc" == k:
                    s += int(len(s)>0)*" | " + "Actn Acc:" + str(accs[k])
                elif "_lang_acc" == k:
                    s += int(len(s)>0)*" | " + "Lang Acc:" + str(accs[k])
            print(s + "\n")
        losses = {k:[v] for k,v in losses.items()}
        accs = {k:[v] for k,v in accs.items()}
        inpts = {**losses, **accs}
        df = pd.DataFrame({k:[v] for k,v in inpts.items()})
        df["epoch"] = epoch
        df["phase"] = self.phase
        df["n_targs"] = n_targs
        df["env_type"] = env_type
        path = os.path.join(
            self.hyps["save_folder"],
            save_name
        )
        header = not os.path.exists(path)
        df.to_csv(
            path,
            sep=",",
            header=header,
            mode="a"
        )

    def collect_data(self, model,
                           env_type,
                           n_targs=None,
                           incl_hs=False,
                           to_cpu=False,
                           n_eps=None,
                           render=False):
        """
        Performs the actual rollouts using the model

        Args:
            model: Module
            env_type: str
            n_targs: int
                the number of targets for the environment to display
            incl_hs: bool
                if true, h vectors are collected from the model
            to_cpu: bool
                if true, the langpreds and actnpreds are placed on
                the cpu
            n_eps: int or None
                the number of episodes to collect
                if None, defaults to hyps['n_eval_eps']
            render: bool
                if true, renders the play. can also be specified
                through hyps.
        Returns:
            data: dict
                keys: str
                vals: shared torch tensors
                    actn_preds: float tensor (N, A)
                        Collects the predictions of the model for each
                        timestep t
                    lang_preds: float tensor (N, L)
                        Collects the predictions of the model for each
                        timestep t
                    actn_targs: long tensor (N,) or float tensor (N,A)
                        Collects the oracle actions at each timestep t.
                        If working with continuous environment, 
                    rews: float tensor (N,)
                        Collects the reward at each timestep t
                    dones: long tensor (N,)
                        Collects the done signals at each timestep t
                    n_targs: long tensor (N,)
                        Collects the number of targets in the episode
                        only relevant if using a gordongames
                        environment variant
                    n_items: long tensor (N,)
                        Collects the number of items over the course of
                        the episode. only relevant if using a
                        gordongames environment variant
                    n_aligned: long tensor (N,)
                        Collects the number of items that are aligned
                        with targets over the course of the episode.
                        only relevant if using a gordongames
                        environment variant
                    incl_hs: bool
                        if true, the returned data includes the hidden
                        states of the model under the keys h0, h1, ...
                        h vectors are collected after the model processes
                        the current state
        """
        data = {
            "states":[],
            "actn_preds":[],
            "lang_preds":[],
            "actn_targs":[],
            "rews":[],
            "dones":[],
            "n_targs":[],
            "n_items":[],
            "n_aligned":[],
            "grabs":[],
            "disp_targs":[],
            "is_animating":[],
            "ep_idx":[],
        }
        state = self.create_new_env(
            n_targs=n_targs,
            env_type=env_type
        )
        model.reset(1)
        if incl_hs: self.record_hs(model=model,data=data)
        ep_count = 0
        if n_eps is None:
            n_eps = try_key(self.hyps,"n_eval_eps",10)
            if self.hyps["exp_name"]=="test": n_eps = 1
        # Get the conditional vector
        with torch.no_grad():
            for i,env_type in enumerate(self.env_types):
                if env_type == self.env.env_type:
                    idxs = model.cdtnl_idxs[i][None]
                    break
            cdtnl = model.cdtnl_lstm(idxs)
        while ep_count < n_eps:
            # Collect the state of the environment
            data["states"].append(state)
            t_state = torch.FloatTensor(state) # (C, H, W)
            # Get action prediction
            inpt = t_state[None].to(DEVICE)
            actn_pred, lang_pred = model.step(inpt, cdtnl)
            data["actn_preds"].append(actn_pred)
            if incl_hs: self.record_hs(model=model,data=data)
            if to_cpu: data["actn_preds"][-1] = actn_pred.cpu()
            # Batch Size is only ever 1
            # lang_pred: (1,1,L)
            if model.n_lang_denses == 1:
                lang = lang_pred[0].unsqueeze(0)
            else:
                lang = torch.stack(lang_pred, dim=0)
            # lang: (N,1,L) where N is number of lang models
            data["lang_preds"].append(lang)
            if to_cpu: data["lang_preds"][-1] = lang.cpu()
            actn = self.get_action(actn_pred)
            # get target action
            targ = self.oracle(self.env)
            data["actn_targs"].append(targ)
            # Step the environment (use oracle if phase 0)
            if self.phase == 0: actn = targ
            obs, rew, done, info = self.env.step(actn)
            state = next_state(
                self.env,
                self.obs_deque,
                obs=obs,
                reset=done,
                n_targs=n_targs
            )
            data["dones"].append(int(done))
            data["rews"].append(rew)
            data["grabs"].append(info["grab"])
            data["ep_idx"].append(self.ep_idx)
            keys = ["n_targs","n_items","n_aligned",
                    "disp_targs","is_animating"]
            for k in keys: data[k].append(info[k])

            if render or self.hyps["render"]:
                self.env.render()
                print("Use count words:", self.hyps["use_count_words"],
                    "-- Lang size:", model.lang_size,
                    "-- N_Targs:", info["n_targs"],
                    "-- N_Items:", info["n_items"]
                )
                lang_targ = get_lang_labels(
                    torch.LongTensor([info["n_items"]]),
                    torch.LongTensor([info["n_targs"]]),
                    max_label=model.lang_size-1,
                    use_count_words=self.hyps["use_count_words"]
                ).item()
                print( "Lang (pred, targ):",
                    torch.argmax(lang.squeeze().cpu().data).item(),
                    "--", lang_targ)
                print("Actn (pred, targ)",
                    actn, "--", data["actn_targs"][-1])
                if done:
                    print( "Actn (pred, targ):",
                        info["n_items"], "--", info["n_targs"])
                    print()
                time.sleep(1)
            if done:
                model.reset(1)
                ep_count += 1
                self.ep_idx += 1
        self.state_bookmark = state
        # Convert data to torch tensors
        # S stands for the collected sequence
        data["actn_preds"] = torch.cat(data["actn_preds"], dim=0) #(S,A)
        data["lang_preds"] = torch.cat(data["lang_preds"], dim=1) #(N,S,L)
        if self.env.is_continuous:
            data["actn_targs"] = torch.from_numpy(
                np.asarray(data["actn_targs"], self.env.action_space.dtype)
            )
        else:
            data["actn_targs"] = torch.LongTensor(data["actn_targs"])
        data["rews"] = torch.FloatTensor(data["rews"])
        keys = ["dones", "grabs", "n_targs", "n_items", "n_aligned",
                "disp_targs", "is_animating", "ep_idx"]
        for k in keys: data[k] = torch.LongTensor(data[k])
        if incl_hs:
            if hasattr(model, "hs"):
                for i in range(len(model.hs)):
                    k = "h"+str(i)
                    data[k] = torch.cat(data[k], dim=0)
            else: data["h0"] = torch.cat(data["h0"], dim=0)
        return data

    def record_hs(self, model, data):
        """
        A function to assist in recording the h vectors when collecting
        data. Operates in place

        Args:
            model: torch Module
            data: dict
        """
        if "h0" not in data:
            if hasattr(model, "hs"):
                for i in range(len(model.hs)):
                    data["h"+str(i)] = []
            else:
                data["h0"] = []
            return
        if hasattr(model, "hs"):
            for i in range(len(model.hs)):
                data["h"+str(i)].append(model.hs[i].cpu().detach().data)
        else:
            data["h0"].append(model.h.cpu().detach().data)
