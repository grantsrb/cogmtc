from cogmtc.experience import DataCollector
import cogmtc.models # SimpleCNN, SimpleLSTM
from cogmtc.recorders import Recorder
from cogmtc.envs import NONVERBAL_TASK_NAMES
from cogmtc.utils.save_io import load_checkpoint
from cogmtc.utils.utils import try_key, get_loss_and_accs
from cogmtc.utils.training import get_resume_checkpt

from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
from tqdm import tqdm
import copy

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def train(rank, hyps, verbose=True):
    """
    This is the main training function. Argue a set of hyperparameters
    and this function will train a model to solve an openai gym task
    given an AI oracle.

    Args:
        rank: int
            the index of the distributed training system.
        hyps: dict
            a dict of hyperparams
            keys: str
            vals: object
        verbose: bool
            determines if the function should print status updates
    """

    # If resuming, hyperparameters are updated appropriately.
    # Actual checkpoint is loaded later.
    _, hyps = get_resume_checkpt(hyps)

    # Set select defaults and seed
    hyps_default_manipulations(hyps)

    # Initialize Data Collector
    # DataCollector's Initializer does Important changes to hyps
    data_collector = DataCollector(hyps)
    # Initialize model
    model = make_model(hyps)
    shared_model = make_model(hyps)
    shared_model.load_state_dict(model.state_dict())
    # Need to delete all non-leaf nodes before we share with other procs
    shared_model.prev_hs = []
    shared_model.prev_cs = []
    shared_model.hs = None
    shared_model.cs = None
    shared_model.h = None
    shared_model.c = None
    shared_model.lang = None
    shared_model.share_memory()
    # Begin collecting data
    if try_key(hyps, "trn_whls_epoch", None) is not None:
        data_collector.init_runner_procs(shared_model)
        print("Sharing model with runners")
    else:
        data_collector.init_runner_procs(None)
    # Set initial phase
    data_collector.update_phase(try_key(hyps, "first_phase", 0))
    # Record experiment settings
    try:
        recorder = Recorder(hyps, model)
    except:
        for k in sorted(hyps):
            print(k, hyps[k])
        assert False
    data_collector.validator.hyps["save_folder"] = hyps["save_folder"]
    data_collector.init_validator_proc(shared_model)
    # initialize trainer
    trainer = Trainer(hyps, model, recorder, verbose=verbose)
    # Skip first phase if true or if first_phase differs from phase
    # of trainer.
    first_phase = try_key(hyps, "first_phase", 0)
    skip_first_phase = try_key(hyps,"skip_first_phase",False)
    hyps_error_catching(hyps)
    if not skip_first_phase and trainer.phase == first_phase:
        s = "\n\nBeginning First Phase: " + str(trainer.phase)
        recorder.write_to_log(s)
        print(s)
        data_collector.dispatch_runners()
        # Loop training
        n_epochs = hyps["lang_epochs"] if first_phase==0 else\
                   hyps["actn_epochs"]
        training_loop(
            n_epochs,
            data_collector,
            trainer,
            model,
            shared_model,
            verbose=verbose
        )
        data_collector.await_runners()
        data_collector.exp_replay.clear_experience()
    # Update phase accross training
    trainer.phase = hyps["second_phase"]
    data_collector.update_phase(trainer.phase)
    data_collector.dispatch_runners()
    # Fresh optimizer
    trainer.set_optimizer_and_scheduler(
        model,
        hyps["optim_type"],
        hyps["lr"],
        try_key(hyps, "resume_folder", None)
    )
    n_epochs = hyps["actn_epochs"] if first_phase==0 else\
               hyps["lang_epochs"]
    s = "\n\nBeginning Second Phase: " + str(trainer.phase)
    recorder.write_to_log(s)
    print(s)
    training_loop(
        n_epochs,
        data_collector,
        trainer,
        model,
        shared_model,
        verbose=verbose
    )
    trainer.end_training(data_collector, shared_model)

def make_model(hyps):
    """
    Makes the model. The model type specified in the hyperparams must
    be imported into the global scope.

    Args:
        hyps: dict
            dict of hyperparameters. See `README.md` for details
    """
    model = cogmtc.models.__dict__[hyps["model_type"]](**hyps).to(DEVICE)
    folder = try_key(hyps, "resume_folder", None)
    init_checkpt = try_key(hyps, "init_checkpt", None)
    lang_checkpt = try_key(hyps, "lang_checkpt", None)
    if folder is not None and folder != "":
        checkpt, _ = get_resume_checkpt(hyps, in_place=False)
        model.load_state_dict(checkpt["state_dict"])
    elif init_checkpt is not None and init_checkpt.strip()!="":
        print("Initializing from checkpoint", init_checkpt)
        checkpt = load_checkpoint(init_checkpt)
        try:
            model.load_state_dict(checkpt["state_dict"])
        except:
            sd = checkpt["state_dict"]
            sd["cdtnl_idxs"] = model.state_dict()["cdtnl_idxs"]
            sd["cnn.cdtnl_idxs"] = model.state_dict()["cnn.cdtnl_idxs"]
            model.load_state_dict(sd)
    elif lang_checkpt is not None and lang_checkpt.strip()!="":
        print("Loading language model", lang_checkpt)
        print("Training will skip to second phase")
        checkpt = load_checkpoint(lang_checkpt, phase=0)
        model.load_state_dict(checkpt["state_dict"])
    return model

def resume_epoch(trainer):
    """
    If the training is resuming from a checkpoint and the phase of the
    checkpoint matches the phase of the trainer, then this function
    returns the epoch of the resumed checkpoint

    Args:
        trainer: Trainer
            a Trainer object. Must have valid hyps member
    Returns:
        epoch: int
            if the phase of the trainer and the checkpoint match, this
            will be the epoch of the checkpoint. In all other cases,
            defaults to 0.
    """
    folder = try_key(trainer.hyps, "resume_folder", "")
    if folder is not None and folder != "":
        checkpt = load_checkpoint(folder)
        if checkpt["phase"] == trainer.phase:
            return try_key(checkpt,"epoch",0)
    return 0

def training_loop(n_epochs,
                  data_collector,
                  trainer,
                  model,
                  shared_model,
                  verbose=True):
    """
    The epoch level training loop.

    Args:
        n_epochs: int
            the number of epochs to train for
        data_collector: DataCollector
        trainer: Trainer
        model: Model
        shared_mode: Model
            deep copy of model with shared weights
    """
    # Potentially modify starting epoch for resumption of previous
    # training. Defaults to 0 if not resuming or not same phase
    start_epoch = resume_epoch(trainer)
    if trainer.hyps["exp_name"]=="test": n_epochs = 2
    for epoch in range(start_epoch, n_epochs):
        if verbose:
            print()
            print("Phase:",
                trainer.phase,
                "-- Epoch",
                epoch,
                "--",
                trainer.hyps["save_folder"]
            )
        # Run environments, automatically fills experience replay's
        # shared_exp tensors
        data_collector.await_runners()
        # Copies the shared exp so that runners can collect
        # during training
        data_collector.exp_replay.harvest_exp(trainer.phase)
        data_collector.dispatch_runners()
        if try_key(trainer.hyps, "incr_seq_len", False):
            max_sl = data_collector.exp_replay.max_seq_len
            min_sl = trainer.hyps["min_seq_len"]
            seqlen = min_sl + epoch * (n_epochs//(max_sl-min_sl))
            data_collector.exp_replay.seq_len = seqlen
        trainer.train(model, data_collector.exp_replay, epoch)

        # Validate Model by Awaiting Validation Process
        if epoch > start_epoch:
            if verbose:
                print("\nAwaiting validator for epoch", epoch-1)
                if type(model) == cogmtc.models.TestModel:
                    print("Len data strings dict:", len(model.data_strings))
                    for k,v in model.data_strings.items():
                        print("v:", v)
            data_collector.await_validator()
            avg_acc = data_collector.val_q.get()
            trainer.save_best_model(shared_model, avg_acc, epoch-1)
        shared_model.load_state_dict(model.state_dict())
        if verbose: print("\nDispatching validator")
        data_collector.dispatch_validator(epoch)

        # Clean up the epoch
        trainer.end_epoch(epoch, n_epochs, model)
    if verbose:
        print("Awaiting validator")
    data_collector.await_validator()

class Trainer:
    """
    This class handles the training of the model.
    """
    def __init__(self, hyps, model, recorder, verbose=True):
        """
        Args:
            hyps: dict
                keys: str
                vals: object
            model: torch.Module
            recorder: Recorder
                an object for recording the details of the experiment
            verbose: bool
                if true, some functions will print updates to the
                console
        """
        self.hyps = hyps
        self.model = model
        self.recorder = recorder
        self.verbose = verbose
        #phase: int [0,1,2]
        #    the phase of the training (is it (0) training the
        #    language network or (1) training the action network or
        #    (2) both together)
        self.phase = self.init_phase()
        self.set_optimizer_and_scheduler(
            self.model,
            self.hyps["optim_type"],
            self.hyps["lr"],
            try_key(self.hyps, "resume_folder", None)
        )
        self.loss_fxn = getattr(F, self.hyps["loss_fxn"])

    def init_phase(self):
        """
        Initializes the phase of the training depending on if the
        training is resuming from a checkpoint.

        Returns:
            phase: int
                either 0 or the phase that the resume model left off at
                during its previous training
        """
        folder = try_key(self.hyps, "resume_folder", "")
        lang_checkpt = try_key(self.hyps, "lang_checkpt", None)
        if folder is not None and folder.strip() != "":
            checkpt = load_checkpoint(folder)
            return try_key(
                checkpt,
                "phase",
                try_key(self.hyps,"first_phase",0)
            )
        # Skip first phase if init from lang_checkpt
        elif lang_checkpt is not None and lang_checkpt.strip()!="":
            return self.hyps["second_phase"]
        return try_key(self.hyps, "first_phase", 0)

    def set_optimizer_and_scheduler(self,
                                    model,
                                    optim_type,
                                    lr,
                                    resume_folder=""):
        """
        Initializes an optimizer using the model parameters and the
        hyperparameters. Also sets a scheduler for the optimizer's
        learning rate.

        If a resume_folder is argued and the phase
        of the resume folder matches the current phase, then the
        optim_dict will be loaded from the resume folder.

        Args:
            model: Model or torch.Module
                any object that implements a `.parameters()` member
                function that returns a sequence of torch.Parameters
            optim_type: str (one of [Adam, RMSprop])
                the type of optimizer. 
            lr: float
                the learning rate
            resume_folder: str (optional)
                a model folder to resume training from. if a valid
                path is argued and the phase of the last checkpt is
                the same as the current phase, the optimizer loads
                thes saved optim_dict
        Returns:
            optim: torch optimizer
                the model optimizer
        """
        self.optim = globals()[optim_type](
            list(model.parameters()),
            lr=lr,
            weight_decay=self.hyps["l2"]
        )
        if resume_folder is not None and resume_folder != "":
            checkpt = load_checkpoint(resume_folder)
            # If same phase, then we want to load the state dict
            # otherwise it means that we originally resumed from phase
            # 0 and now we're past that so we won't want to load the sd
            if try_key(checkpt, "phase", None) == self.phase:
                self.optim.load_state_dict(checkpt["optim_dict"])
            elif try_key(checkpt["stats"],"phase",None) == self.phase:
                self.optim.load_state_dict(checkpt["optim_dict"])

        self.scheduler = ReduceLROnPlateau(
            self.optim,
            mode='min',
            factor=try_key(self.hyps,"factor", 0.9),
            patience=try_key(self.hyps, "patience", 10),
            threshold=try_key(self.hyps, "threshold", 0.0001),
            min_lr=try_key(self.hyps, "min_lr", 0),
            verbose=self.verbose
        )

    def reset_model(self, model, batch_size):
        """
        Determines what type of reset to do. If the data is provided
        in a random order, the model is simply reset. If, however,
        the data is provided in sequence, we must store the h value
        from the first forward loop in the last training loop.
        """
        if self.hyps["randomize_order"]:
            model.reset(batch_size=batch_size)
        elif try_key(self.hyps, "roll_data", True):
            model.reset_to_step(step=0)
        else:
            model.reset_to_step(step=-1)

    def train(self, model, data_iter, epoch):
        """
        This function handles the actual training. It loops through the
        available data from the experience replay to train the model.

        Args:
            model: torch.Module
                the model to be trained
            data_iter: iterable
                an iterable of the collected experience/data. each 
                iteration must return a dict of data with the keys:
                    obs: torch Float Tensor (N, S, C, H, W)
                    actns: torch Long Tensor (N,S)
                    dones: torch Long Tensor (N,S)
                    n_targs: None or torch LongTensor (N,S)
                The iter must also implement the __len__ member so that
                the data can be easily looped through.
        """
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        model.train()
        model.reset(self.hyps['batch_size'])
        for i,data in enumerate(data_iter):
            iter_start = time.time()
            self.optim.zero_grad()
            obs =   data["obs"]
            actns = data["actns"]
            dones = data["dones"]
            drops = data["drops"]
            n_items = data["n_items"]
            n_targs = data["n_targs"]
            labels = data["lang_labels"]
            tasks = data["tasks"]
            masks = data["masks"]

            if self.phase == 0 and try_key(self.hyps,"blind_lang",False):
                obs = torch.zeros_like(obs)
                dones = torch.zeros_like(dones)
                reps = self.hyps["lang_range"][1]*2
                if i % reps == reps-1:
                    dones[:,0] = 1


            # Testing
            #############
            if epoch > -1 and self.hyps["exp_name"]=="test" and\
                            try_key(self.hyps,"static_render",False):
                grabs = data["grabs"]
                #print("train grabs:")
                #for row in range(len(drops)):
                #    print(grabs[row].cpu().numpy())
                #print("train n_targs:")
                #for row in range(len(drops)):
                #    print(n_targs[row].cpu().numpy())
                #print("train n_items:")
                #for row in range(len(drops)):
                #    print(n_items[row].cpu().numpy())
                #print("train drops:")
                #for row in range(len(drops)):
                #    print(drops[row].cpu().numpy())
                #print("lang labels:")
                #for row in range(len(drops)):
                #    print(labels[row].cpu().numpy())
                #print("train actns:")
                #for row in range(len(drops)):
                #    print(actns[row].cpu().numpy())

                print("Starting new loop")
                #o = obs.detach().cpu().data.numpy()
                #o = o[:,:, 0].transpose((0,2,1,3)).reshape(-1, o.shape[2]*o.shape[1])
                #fig = plt.figure(figsize=(10,10))
                #plt.imshow(o)
                #plt.savefig("imgs/epoch{}_iter{}.png".format(epoch, i))
                ##plt.show()
                for row in range(min(len(obs),5)):
                    print("row:",row)
                    for ii,o in enumerate(obs[row].detach().cpu().numpy()):
                        print("seq:", ii)
                        print("n_items:", n_items[row,ii].cpu().numpy())
                        print("n_targs:", n_targs[row,ii].cpu().numpy())
                        print("drops:", drops[row,ii].cpu().numpy())
                        print("labels:", labels[row,ii].cpu().numpy())
                        print("actns:", actns[row,ii].cpu().numpy())
                        print("isanim:", data["is_animating"][row,ii].cpu().numpy())
                        print("masks:", masks[row,ii].cpu().numpy())
                        print("tasks:", tasks[row,ii].cpu().numpy())
                        print()
                        time.sleep(1)
                        #plt.imshow(o.transpose((1,2,0)).squeeze())
                        #plt.show()
                ##        #plt.savefig("imgs/epoch{}_row{}_samp{}.png".format(epoch, row, ii))
            ###############

            # Resets to h value to appropriate step of last loop
            self.reset_model(model, len(obs))

            # If no drops exist during phase 0, no point in carrying out
            # the backprop loop
            if (drops.sum() == 0 and self.phase != 1) \
                                or masks.float().sum() == len(masks):
                print("No drops in loop", i, "... continuing")
                with torch.no_grad():
                    logits, langs = model(
                        obs.to(DEVICE),
                        dones=dones.to(DEVICE),
                        tasks=tasks.to(DEVICE),
                        masks=masks.to(DEVICE)
                    )
                continue

            inps = None
            if try_key(self.hyps, "incl_lang_inpt", False):
                # No teacher forcing by probability p
                p = try_key(self.hyps, "lang_teacher_p", 0.9)
                if np.random.random() < p:
                    # Set initial label to be STOP
                    s = labels.shape
                    if len(s)==3:
                        stop_label = torch.zeros(s[0],1,s[2])
                        stop_label[:,:,0] = self.hyps["STOP"]
                    else:
                        stop_label=torch.zeros(s[0],1)+self.hyps["STOP"]
                    inps = torch.cat([stop_label,labels[:,:-1]],dim=1)
                    inps = inps.to(DEVICE).long()
                    if try_key(self.hyps,"shuffle_lang_inpts",False):
                        s = inps.shape
                        perm = torch.randperm(int(np.prod(s))).long()
                        inps = inps.reshape(-1)[perm].reshape(s)

            # model uses dones if it is recurrent
            logits, langs = model(
                obs.to(DEVICE),
                dones=dones.to(DEVICE),
                tasks=tasks.to(DEVICE),
                masks=masks.to(DEVICE),
                lang_inpts=inps
            )

            loss, losses, accs = get_loss_and_accs(
                phase=self.phase,
                loss_fxn=self.loss_fxn,
                actn_preds=logits,
                lang_preds=langs,
                actn_targs=actns,
                lang_targs=labels,
                masks=masks,
                drops=drops,
                n_targs=n_targs,
                n_items=n_items,
                prepender="train",
                lang_p=self.hyps["lang_p"],
                # this only matters if using NUMERAL system
                lang_size=self.hyps["lang_size"],
                # this only matters if using NUMERAL system
                use_count_words=self.hyps["use_count_words"],
                null_alpha=try_key(self.hyps, "null_alpha", 0.1)
            )
            # Backprop and update
            loss.backward()
            if try_key(self.hyps, "grad_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.hyps["grad_norm"]
                )
            if try_key(self.hyps, "drop_grad", 0) > 0:
                drop_p = self.hyps["drop_grad"]
                for p in model.parameters():
                    if hasattr(p.grad, "data"):
                        drop_mask = torch.rand_like(p.grad.data)<drop_p
                        p.grad.data[drop_mask] = 0
            self.optim.step()
            # Calc acc
            # Record metrics
            metrics = {
                "train_loss": loss.item(),
                **losses,
                **accs}
            self.recorder.track_loop(metrics)
            key = "train_lang_acc" if self.phase==0 else "train_actn_acc"
            self.print_loop(
                i,
                len(data_iter),
                loss.item(),
                try_key(accs,key,try_key(losses,"train_actn_loss",None)),
                iter_start
            )
            if self.hyps["exp_name"] == "test" and i >= 2: break
        key = "train_lang_loss" if self.phase==0 else "train_actn_loss"
        self.scheduler.step(
            np.mean(self.recorder.metrics[key])
        )

    def print_loop(self,
                   loop_count,
                   max_loops,
                   loss,
                   acc,
                   iter_start):
        """
        Printing statement for inner loop in the epoch.

        Args:
            loop_count: int
                the current loop
            max_loops: int
                the number of loops in the epoch
            loss: float
                the calculated loss
            acc: float
                the calculated accuracy
            iter_start: float
                a timestamp collected at the start of the loop
        """
        s = "Loss:{:.5f} | Acc:{:.5f} | {:.0f}% | t:{:.2f}"
        s = s.format(
            loss,
            acc,
            loop_count/max_loops*100,
            time.time()-iter_start
        )
        print(s, end=len(s)//4*" " + "\r")

    def calc_targ_accs(self,
        n_targs,
        n_items,
        n_aligned,
        prepender="val",
        **kwargs
    ):
        """
        Calculates the accuracy of the episodes with regards to matching
        the correct number of objects.

        Args:
            n_targs: ndarray or long tensor (N,)
                Collects the number of targets in the episode
                only relevant if using a gordongames
                environment variant
            n_items: ndarray or long tensor (N,)
                Collects the number of items over the course of
                the episode. only relevant if using a
                gordongames environment variant
            n_aligned: ndarray or long tensor (N,)
                Collects the number of items that are aligned
                with targets over the course of the episode.
                only relevant if using a gordongames
                environment variant
            prepender: str
                a simple string prepended to each key in the returned
                dict
        Returns:
            metrics: dict
                keys: str
                    "error": float
                        the difference between the number of target
                        objects and the number of item objects
                    "coef_of_var": float
                        the coefficient of variation. The avg error
                        divided by the goal size
                    "stddev": float
                        the standard deviation of the n_item responses.
                    "mean_resp": float
                        the mean response of the n_item responses.
        """
        fxns = {
            "error": calc_error,
            "coef_of_var": coef_of_var,
            "stddev": stddev,
            "mean_resp": mean_resp,
        }
        metrics = dict()
        if type(n_targs) == torch.Tensor:
            n_targs = n_targs.detach().cpu().numpy()
        if type(n_items) == torch.Tensor:
            n_items = n_items.detach().cpu().numpy()
        if type(n_aligned) == torch.Tensor:
            n_aligned = n_aligned.detach().cpu().numpy()
        inpts = {
            "n_items":  n_items,
            "n_targs":  n_targs,
            "n_aligned":n_aligned,
        }
        categories = set(n_targs.astype(np.int))
        for key,fxn in fxns.items():
            metrics[prepender+"_"+ key] = fxn(**inpts)
            # Calc for each specific target count
            for cat in categories:
                targs = n_targs[n_targs==cat]
                items = n_items[n_targs==cat]
                aligned = n_aligned[n_targs==cat]
                if len(targs)==0 or len(items)==0 or len(aligned)==0:
                    continue
                metrics[prepender+"_"+key+"_"+str(cat)] = fxn(
                    n_items=items,
                    n_targs=targs,
                    n_aligned=aligned,
                )
        return metrics

    def save_best_model(self, model, score, epoch):
        """
        If accuracy is best yet saves the model as the best model.
        Only use for phases 1 and 2 

        Args:
            model: torch Module
            score: float
                validation accuracy or some other score where greater
                is better.
            epoch: int
                the epoch associated with the model
        """
        if self.phase == 0: return # the score is generally the actn acc
        self.recorder.save_best_model(model, score, epoch, self.phase)

    def end_epoch(self, epoch, n_epochs, model):
        """
        Records, prints, cleans up the epoch statistics. Also handles
        the training wheels probabilities. Call this function at the
        end of an epoch.

        Args:
            epoch: int
                the epoch that has just finished.
            n_epochs: int
                the total epoch count
        """
        self.recorder.save_epoch_stats(
            self.phase,
            epoch,
            self.model,
            self.optim,
            verbose=self.verbose
        )
        self.recorder.reset_stats()

        trn_whls = try_key(self.hyps, "trn_whls_epoch", None)
        trn_whls_off = trn_whls is not None and epoch >= trn_whls
        if self.phase != 0 and trn_whls_off:
            model.trn_whls = self.update_trn_whls_p(epoch, n_epochs)

    def update_trn_whls_p(self,epoch, n_epochs):
        """
        Updates the training wheels probability where trn_whls_p is
        the probability of using the oracle to pick an action and 
        1-trn_whls_p is the probability of using the agent to select
        an action.. The trn_whls_p is linearly decreased from its
        initial value to the trn_whls_min value over the course of
        the trn_whls_epoch to the final epoch.

        Args:
            epoch: int
                the epoch that has just finished.
            n_epochs: int
                the total epoch count
        """
        p = self.hyps["trn_whls_p"]
        whls_epoch = self.hyps["trn_whls_epoch"]
        progress = float(epoch-whls_epoch)/float(n_epochs-whls_epoch)
        whls_min = self.hyps["trn_whls_min"]
        return max(p - (p-whls_min)*progress, whls_min)

    def end_training(self, data_collector, shared_model):
        """
        Perform all cleanup actions here. Mainly recording the best
        metrics.
        
        Args:
            data_collector: DataCollector
            shared_model: shared torch nn Module
        """
        data_collector.terminate_procs()
        keys = list(data_collector.exp_replay.shared_exp.keys())
        for k in keys:
            t = data_collector.exp_replay.shared_exp[k]
            del t
            del data_collector.exp_replay.shared_exp[k]
        del shared_model
        del data_collector.exp_replay
        del data_collector.gate_q
        del data_collector.stop_q
        del data_collector.val_gate_q
        del data_collector.val_stop_q
        del data_collector.phase_q
        del data_collector.terminate_q
        del data_collector

def mean_resp(n_items, **kwargs):
    """
    Args:
        n_items: ndarray (same dims as n_targs)
    Returns:
        mean: float
            the standard deviation of the responses
    """
    return n_items.mean()

def stddev(n_items, **kwargs):
    """
    Args:
        n_items: ndarray (same dims as n_targs)
    Returns:
        std: float
            the standard deviation of the responses
    """
    return n_items.std()

def calc_error(n_items, n_targs, **kwargs):
    """
    The square root of the mean squared distance between n_items and 
    n_targs.

    Args:
        n_items: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_items)
    Returns:
        error: float
            the square root of the average squared distance from the
            goal.
    """
    return np.sqrt(((n_items-n_targs)**2).mean())

def coef_of_var(n_items, n_targs, **kwargs):
    """
    Returns the coefficient of variation which is the error divided
    by the average n_targs

    Args:
        n_items: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_items)
    Returns:
        coef_var: float
            the error divided by the average n_targs
    """
    if len(n_items) == 0: return np.inf
    mean = n_items.mean()
    if mean == 0: return np.inf
    return n_items.std()/n_items.mean()

def perc_aligned(n_aligned, n_targs, **kwargs):
    """
    Calculates the percent of items that are aligned

    Args:
        n_aligned: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_aligned)
    Returns:
        perc: float
            the average percent aligned over all entries
    """
    perc = n_aligned/n_targs
    return perc.mean()*100

def perc_unaligned(n_items, n_aligned, n_targs, **kwargs):
    """
    Calculates the percent of items that are unaligned

    Args:
        n_items: ndarray (same dims as n_targs)
        n_aligned: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_items)
    Returns:
        perc: float
            the average percent unaligned over all entries
    """
    perc = (n_items-n_aligned)/n_targs
    return perc.mean()*100

def perc_over(n_items, n_targs, **kwargs):
    """
    Calculates the average proportion in which the number of items
    was greater than the number of targets. If the number of items
    was less than or equal to the number of targets, that entry is
    counted as 0%

    Args:
        n_items: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_items)
    Returns:
        perc: float
            the average amount of items over the number of targets
    """
    n_items = n_items.copy()
    n_items[n_items<n_targs] = n_targs[n_items<n_targs]
    perc = (n_items-n_targs)/n_targs
    return perc.mean()*100

def perc_under(n_items, n_targs, **kwargs):
    """
    Calculates the average proportion in which the number of items
    was less than the number of targets. If the number of items
    was greater than or equal to the number of targets, that entry is
    counted as 0%

    Args:
        n_items: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_items)
    Returns:
        perc: float
            the average amount of items less than the number of targets
    """
    n_items = n_items.copy()
    n_items[n_items>n_targs] = n_targs[n_items>n_targs]
    perc = (n_targs-n_items)/n_targs
    return perc.mean()*100

def perc_off(n_items, n_targs, **kwargs):
    """
    Calculates the average proportion in which the number of items
    was different than the number of targets.

    Args:
        n_items: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_items)
    Returns:
        perc: float
            the average amount of items different than the number of
            targets
    """
    perc = torch.abs(n_targs-n_items)/n_targs
    return perc.mean()*100

def perc_correct(n_aligned, n_targs, **kwargs):
    """
    Calculates the average proportion in which the number of aligned
    items is equal to the number of targets.

    Args:
        n_aligned: ndarray (same dims as n_targs)
        n_targs: ndarray (same dims as n_aligned)
    Returns:
        perc: float
            the average number of entries in which the number of
            aligned items is equal to the number of targets.
    """
    perc = (n_aligned == n_targs)
    return perc.mean()*100

def hyps_default_manipulations(hyps):
    """
    Here we can do some manipulations to ensure select defaults are
    correct.
    """
    # Set random seeds
    hyps['seed'] = try_key(hyps,'seed', int(time.time()))
    seed = hyps["seed"]
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Handle Hold Outs
    if "hold_out" in hyps and "hold_outs" not in hyps:
        hyps["hold_outs"] = hyps["hold_out"]
        del hyps["hold_out"]
        print("changing hyps key 'hold_out' to 'hold_outs'")
    hyps["hold_outs"] = try_key(hyps, "hold_outs", None)
    if hyps["hold_outs"] is not None and len(hyps["hold_outs"])==0:
        hyps["hold_outs"] = None

    # Hold outs are used in the get_drops function as a way
    # to determine if a word should ever be trained on. 
    # The hold_lang default to hold_outs if nothing is argued.
    # Please forgive my blatant typing abuse and abusive overextension
    # of the use of None
    assert "hold_words" not in hyps, "You probably meant hold_lang"
    assert "hold_langs" not in hyps, "You probably meant hold_lang"
    hyps["hold_lang"] = try_key(hyps, "hold_lang", True)
    if hyps["hold_lang"] is True:
        hyps["hold_lang"] = hyps["hold_outs"]
    elif hyps["hold_lang"] is False or hyps["hold_lang"] is None:
        hyps["hold_lang"] = None
    else: hyps["hold_lang"] = set(hyps["hold_lang"])

    # The hold_actns default to hold_outs if nothing is argued.
    # hold_actns determines what final target counts will be used in game
    # Please forgive my blatant typing abuse and abusive overextension
    # of the use of None
    hyps["hold_actns"] = try_key(hyps, "hold_actns", True)
    if hyps["hold_actns"] is True:
        hyps["hold_actns"] = hyps["hold_outs"]
    elif hyps["hold_actns"] is False or hyps["hold_actns"] is None:
        hyps["hold_actns"] = None
    else: hyps["hold_actns"] = set(hyps["hold_actns"])

def hyps_error_catching(hyps):
    """
    Here we can check that the hyperparameter configuration makes sense.
    """
    if try_key(hyps,"blind_lang",False):
        print("Setting langall to true for blind lang")
        hyps["langall"] = True
        if try_key(hyps,"drop_perc_threshold",0)!=0:
            print("changing drop_perc_threshold to 0")
            hyps["drop_perc_threshold"] = 0

    if "fc_norm" in hyps:
        print("found fc_norm in hyps, changing to fc_lnorm")
        hyps["fc_lnorm"] = hyps["fc_norm"]

    msl = try_key(hyps, "min_seq_len", None)
    if msl is None or msl <= 1:
        default_val = 7
        print("Setting min_seq_len to", default_val)
        hyps["min_seq_len"] = default_val

    adder = hyps["grid_size"][-1]
    max_steps = hyps["val_targ_range"][-1]*2 + adder
    hyps_steps = try_key(hyps, "max_steps", 0)
    if max_steps > hyps_steps:
        hyps["max_steps"] = max_steps
        print("Found impossible max_steps, changing to", max_steps)

    # Convert to No-Language Variant if -1 is argued
    if hyps["use_count_words"]==-1:
        hyps["use_count_words"] = 1
        hyps["second_phase"] = 1
        hyps["skip_first_phase"] = True
        hyps["incl_lang_inpt"] = False
        hyps["incl_actn_inpt"] = False

    if "langall" not in hyps and "lang_on_drops_only" in hyps:
        hyps["langall"] = not hyps["lang_on_drops_only"]
    if try_key(hyps, "lang_targs_only", False) and\
            try_key(hyps,"langall",False):
        print("Potential conflict between lang_targs_only and langall")
        print("langall takes precedence. lang will occur at all steps")
    # Defaults here avoid errors in the future
    hyps["count_targs"] = True
    print("Setting count_targs to true")
    #if hyps["use_count_words"] == 0:
    #    hyps["count_targs"] = False
    #    print("Setting count_targs to false for inequality labels")
    #else:
    #    hyps["count_targs"] = True
    #    print("Setting count_targs to true")
    hyps["n_envs"] = len(hyps["env_types"])
    if hyps["batch_size"] % hyps["n_envs"] != 0:
        print(
            "Batch size of", hyps["batch_size"],
            "must be divisible by the number of envs", hyps["n_envs"]
        )
        hyps["batch_size"] = (hyps["batch_size"]//hyps["n_envs"])*hyps["n_envs"]
        print("Changing batch_size to", hyps["batch_size"])

    for env_type in hyps["env_types"]:
        if env_type in NONVERBAL_TASK_NAMES:
            hyps["targ_range"] = [2,2]
            hyps["val_targ_range"] = [1,3]
            print("Non verbal task, setting targ range to", hyps["targ_range"])
            break
    if "grad_clip" in hyps:
        hyps["grad_norm"] = hyps["grad_clip"]
        print("grad_clip found in hyps, changing to grad_norm")
    if "incl_lang_inpts" in hyps and "incl_lang_inpt" not in hyps:
        hyps["incl_lang_inpt"] = hyps["incl_lang_inpts"]
        print("Fixing plural mistake, incl_lang_inpts to incl_lang_inpt")
    if "lstm_lang_inpt" in hyps and "incl_lang_inpt" not in hyps:
        hyps["incl_lang_inpt"] = hyps["lstm_lang_inpt"]
        print("Fixing naming mistake, lstm_lang_inpt to incl_lang_inpt")
    if "incl_actn_inpts" in hyps and "incl_actn_inpt" not in hyps:
        hyps["incl_actn_inpt"] = hyps["incl_actn_inpts"]
        print("Fixing plural mistake, incl_actn_inpts to incl_actn_inpt")
    if "lstm_actn_inpt" in hyps and "incl_actn_inpt" not in hyps:
        hyps["incl_actn_inpt"] = hyps["lstm_actn_inpt"]
        print("Fixing naming mistake, lstm_actn_inpt to incl_actn_inpt")

