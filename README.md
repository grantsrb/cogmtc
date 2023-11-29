# Conditional Grounded Multi-Task Counting (COGMTC)

## Description
This project seeks to disentangle the reason why language is correlated with some cognitive tasks. Specifically, this project seeks to determine if language is used as a tool for counting or it acts as a practice for learning counting concepts.

The experiment first creates two models. One is pretrained to use language to count objects. The other is pretrained on a task that requires counting but only uses a trinary signal of more, less, or equal to as its output. The models are then transferred to tasks that require manipulating an environment based on a visual count. If the two models perform equally, we can attribute the cognitive abilities associated with language to the practice that comes with learning a language. If the lingual model performs better, however, we can attribute the improvements to the model using language as a tool.

## How to Use this Repo
### Training
To train a model you will need to have a hyperparameters json and a hyperranges json. The hyperparameters json details the values of each of the training parameters that will be used for the training. See the [training scripts readme](training_scripts/readme.md) for parameter details. The hyperranges json contains a subset of the hyperparameter keys each coupled to a list of values that will be cycled through for training. Every combination of the hyperranges key value pairs will be scheduled for training. This allows for easy hyperparameter searches. For example, if `lr` is the only key in the hyperranges json, then trainings for each listed value of the learning rate will be queued and processed in order. If `lr` and `l2` each are in the hyperranges json, then every combination of the `lr` and `l2` values will be queued for training.

To run a training session, navigate to the `training_scripts` folder:

```
$ cd training_scripts
```

And then select the cuda device index you will want to use (in this case 0) and type the following command:

```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py path_to_hyperparameters.json path_to_hyperranges.json
```
It is also possible to split the hyperranges into multiple tmux sessions on multiple gpus using the `./training_scripts/distr_main.py` script. To use this script, you must create a metaparams.json file which is detailed within the `distr_main.py` script. If it does not work, you may need to upgrade tmux to at least 3.0 or you may need to upgrade ubuntu to 20.04 or later.

## Setup
After cloning the repo, install all necessary packages locally:

```sh
python3 -m pip install --user -r requirements.txt
```

We use submodules for the gordongames environments. You will need to
initialize these doing the following.

```sh
cd gordongames
git submodule init
git submodule update
```

Next you will to install this pip package and the gordongames package.
Run the following in both the gordongames directory and the top
directory of this project. The following assumes you're still located
in the gordongames directory from the previous chunk of commands:

```sh
python3 -m pip install --user -e .
cd ..
python3 -m pip install --user -e .
```

## Repo Usage
### Watching Your Trained Policy
After training your policy, you can watch the policy run in the environment using the `watch_model.py` script. To use this file, pass the name of the saved model folder that you would like to watch. The viewing script will automatically create a version of the environment that the model was trained on, load the best version of the model based on the evaluated performance during training, and run the model on the environment.

Here's an example:

    $ python watch_model.py search_name/search_name_0_whateverwaslistedhere/

### Automated Hyper Parameter Search
Much of deep learning consists of tuning hyperparameters. It can be extremely addicting to change the hyperparameters by hand and then stare at the average reward as the algorithm trains. THIS IS A HOMERIAN SIREN! DO NOT SUCCUMB TO THE PLEASURE! It is too easy to change hyperparameters before their results are fully known. It is difficult to keep track of what you did, and the time spent toying with hyperparameters can be spent reading papers, studying something useful, or calling your Mom and telling her that you love her (you should do that right now. Your dad, too)

This repo can automate your hyperparameter searches using a `hyperranges json`. Simply specify the key you would like to search over and specify a list of values that you would like that key to take. If multiple keys are listed, then all combinations of the possible values will be searched. 

### Saved Outputs
Each combination of hyperparameters gets saved to a "model\_folder" under the path within a greater folder called the "exp\_folder". The experiment folder is specified within the `hyperparams.json`.

Within the `model_folder` a checkpoint file exists for each epoch in the training. The checkpoints contain details about the statistics for that epoch as well as holding the hyperparameters for that epoch. The "best" and "last" checkpoints also contain the model parameters. You can save a copy of the model parameters at every epoch using the `hyperparams.json`.

### `validation\_stats.csv`
For this project a csv is created for each training session. The csv consists of the ending metrics of each validation episode for each epoch. This should enable you to analyze how the model performance changed over the course of the training and allow you to calculate statistics such as the coefficient of variation.


#### List of Valid Keys for hyperparams json
Set values in a json and run `$ python3 main.py your_hyperparams_json.json` to use the specified parameters.

    "exp_name": str
        the name of the folder in which the hyperparameter search will
        be saved to. This is different than the path. If you would like
        to save the experiment to a different folder than the one in
        which you run `main.py`, use the hyperparemter called `save_root`
    "exp_num_offset": int
        a number by which to offset all experiment numbers. This is
        useful in cases where you want to run trainings on different
        machines but avoid overlapping experiment numbers
    "save_root": str
        this value is prepended to the exp_name when creating the save
        folder for the hyperparameter search.
    "resume_folder": str
        path to a training to resume from. The epochs argued in this
        hyperparameter set must be larger than where the resumed
        training left off.
    "init_checkpt": str
        path to a model checkpoint to initialize the model from.
        If a model_folder is argued (instead of a specific checkpoint
        file), the loaded weights will be from the last epoch of the
        training.
    "lang_checkpt": str
        path to a language checkpoint to initialize the model from.
        If a model_folder is argued (instead of a specific checkpoint
        file), the loaded weights will be from the last epoch of phase
        0. This makes it possible to save time by using pretrained
        language models. This will make the training skip straight to
        the second phase but will leave the `skip_first_phase` parameter
        unchanged.
    "description": str
        this is an option key used to write notes or a description of
        the hyperparameter search
    "render": bool
        if true, the validation operations are rendered
    "del_prev_sd": bool
        option to only keep the most recent state dict in the
        checkpoints. This is used to conserve space on the disk. If
        true, previous state dicts are deleted from their checkpts
    "seed": int
        The random seed for the training
    "runner_seed_offset": int
        An offset for the random seed for each of the parallel
        environments
    "pre_epochs": int
        the number of epochs used during pretraining
    "n_epochs": int
        the number of epochs used for the final phase of the
        training.
    "lang_epochs": int (deprecated for `pre_epochs` and `n_epochs`)
        The number of training epochs for the language training phase
    "actn_epochs": int (deprecated for `pre_epochs` and `n_epochs`)
        The number of training epochs for the action training phase

    "trn_whls_epoch": int
        the epoch in which the training wheels are removed from the
        model. This means that the training data is collected using
        the actions from the model. The action targets continue to be
        those produced by the oracle.
    "trn_whls_p": float
        the initial probability of using an action from the oracle to
        step the environment after the `trn_whls_epoch`. 1 means that
        all actions used to step the environment are taken from the
        oracle. This value is automatically linearly interpolated from
        its initial value to the `trn_whls_min` over the course of
        `n_epochs`-`trn_whls_epoch`. If `trn_whls_min` is set to the
        same value as `trn_whls_p` then no decaying takes place.
    "trn_whls_min": float
        this is the minimum value of the trn_whls_p. see the description
        of `trn_whls_p` for a explanation of how this value is used in
        training.

    "use_count_words": int [0, 1, 2, 3, 4, or 5]
        this determines what type of count words should be used. If 0,
        the model learns to count by issuing a less-than, equal-to, or
        greater-than prediction of the number of items relative to
        the number of targets. If 1, the model learns to count out the
        items using english number words. Lastly if 2 is argued, the
        model learns the Piraha count words which are probabilistically
        derived at runtime to match the statistics found in Frank 2008.
        In this case, 4 possible labels exist. 0, 1, and 2 each are used
        100% of the time for numbers 0,1,and 2. But for numbers 3 and
        greater, either label 2 or 3 is used for the count word with
        a probability matching the published values.

        -1: no language labels
        0: inequality labels
        1: english labels
        2: piraha labels
        3: random labels
        4: two equivalent labels exist for every label
        5: numeral system with argued base `numeral_base`
    "actnlish": bool
        if true, the language model is trained to predict the action
        label for game steps in which the player has responded the
        correct number of times after the animation phase and/or the
        player is no longer on the pile. Overrides `langall` to true.
        Will ignore nullese if actnlish is true.
    "nullese": bool
        if true, the language model is trained to predict a null label
        for game steps in which the player has responded the
        correct number of times after the animation phase and/or the
        player is no longer on the pile. Will set `langall` to true.
        will be ignored if actnlish is true.
    "skippan": bool
        if true, will include a language prediction for skipped steps
        in random timing trials. Otherwise defaults to repeating the
        current count during skipped steps.
    "skip_is_null": bool
        if true, will set the skip token (used in the `skippan` setting)
        equal to the null token to conserve the number of language
        prediction types. If false, creates an entirely new language
        token for skipped frames. Only applies if `skippan` is true.
    "numeral_base": int or None
        if base is argued and use_count_words is 5, lang_preds are
        sequential and are trained to output the numerals inline with
        the argued base instead of one-hot labels.
    "rev_num": bool
        if true, the InptConsolidationModule will reverse the
        order of numerals up to the STOP token when processesing
        language inputs. for example, the array [1,2,3,STOP]
        will become [3,2,1,STOP]. only operates along the last dimension,
        only applies to NUMERAL model types
    "skip_first_phase": bool
        if true, the training will skip phase 0 and go straight to
        phase 1 or 2 (depending on the value of `second_phase`).
        Defaults to false.
    "pre_rand": bool
        if true, uses random labels for language pretrainings. Currently
        only implemented for English count words. Only applies during
        phase 0 on the first phase.
    "first_phase": int (0, 1 or 2)
        this determines if the model will be trained with language,
        actions, or both during the first phase of the training.
        0 means language only. 1 means the model is trained with actions
        only. 2 means the model is trained with both language and
        actions.
    "second_phase": int (0, 1 or 2)
        this determines if the model will be trained with language,
        actions, or both during the second phase of the training.
        0 means language only. 1 means actions only. 2 means both
        language and actions.
    "blind_lang": bool
        if true, the observations are zeroed out during language 
        training, creating the effect of the model purely learning
        language. If false, this variable is effectively ignorned.
        Defaults to False.

    "model_type": str
        the name of the model class that you wish to use for the
        training. i.e. "SimpleCNN"
    "sep_pathways": bool
        if true, will create the model to have separate language and
        action pathways (separate defined by no shared computation and
        thus no shared gradients). The language prediction, however,
        will still be used by the action network.
    "splt_feats": bool
        effectively creates a separate convolutional network
        for the language pathway in the NSepLSTM variants.
        This ensures that the language and policy pathways do
        not overlap at all.
    "aux_lang": bool
        if true, will set `incl_lang_inpt` to false. The goal is to
        ensure that a language prediction is made, but is not included
        in the action lstm pathway 
    "n_lstms": int
        The number of LSTMs to use in the model type. This only applies
        for the NVaryLSTM and NSepLSTM variants. In the NVaryLSTM, the
        `n_lstms` arg determines how many LSTMs are chained together for
        the policy network. The language output is either off the
        output from the first or last LSTM in the chain. For the
        NSepLSTM model type, `n_lstms` creates a chain of `n_lstms`-1
        LSTMs for the policy network and uses 1 LSTM for the language
        prediction which is used to select an embedding as additional
        input into the policy network's first LSTM.

        NOTE: See n_pre_lstms, n_lang_lstms, and n_actn_lstms for 
        PreNSepLSTM model types.
    "n_pre_lstms": int
        determines the number of lstms directly following the
        visual latent vector for PreNSepLSTM model types
    "n_lang_lstms": int
        determines the number of LSTMs in the language pathway
        following the pre pathway for PreNSepLSTM model types
    "n_actn_lstms": int
        determines the number of LSTMs in the actn pathway
        following the pre pathway for PreNSepLSTM model types
    "lstm_lang": bool
        if true, and using numeral system, the language output will be
        done using an LSTM rather than a dense network. only applies
        for NUMERAL systems.
    "lstm_lang_first": bool
        only used in multi-lstm model types. If true, the h
        vector from the first LSTM will be used as the input
        to the language layers. The second h vector will then
        be used for the action layers. If False, the second h
        vector will be used for language and the first h for
        actions. Defaults to True. For transformers, this determines
        if the language tokens or vision tokens come first when
        alternating within context. Only applies in transformersif
        stack_context is false.
    "stagger_preds": bool
        if true, the language and action predictions are made from
        different LSTMs within the model if possible. The order is
        determined using `lstm_lang_first`
    "incl_lang_inpt": bool
        if true, the softmax or one-hot
        encoding or h vector (depending on `langactn_inpt_type`)
        of the language output for the current timestep is included as
        input into the action prediction. If false,
        the language output is not included.
    "emb_ffn": bool
        if true, will apply a feed forward network on any language
        embeddings used as input to the policy network. only applies
        when `incl_lang_inpt` is true.
    "one_hot_embs": bool
        only applies when using `incl_lang_inpt`. will ensure that all
        language embeddings used as input to the policy lstms are
        one-hot encodings if true.
    "lang_inpt_layer": int
        index of the lstm within the policy pathway to feed the lang
        prediction. 0 means the earliest lstm, this is the default.
        -1 means the lstm just before the action prediction. Only
        applicable in NSepLSTM variants.
    "cut_lang_grad": bool
        if true, the gradient from the language pathway is not
        propagated beyond the first language lstm
    "extra_lang_pred": bool
        if true, the DoubleVaryLSTM will include an extra
        language prediction system that it will use for making
        language predictions in the `incl_lang_inpt` cases
        (emulating the behavior of the SeparateLSTM). This is
        in addition to the DoubleVaryLSTM's usual
        language predictions. Only relevant when using the
        DoubleVaryLSTM model type and `incl_lang_preds` is true.
    "tforce": bool
        if true, allows teacher forcing. See `lang_teacher_p` to make
        teacher forcing probabilistic during training. Setting tforce
        to true will always use teacher forcing during both training
        and validation.
    "tforce_train": bool
        overwritten by `tforce`. If true, will use teacher forcing
        during training. Makes no changes to teacher forcing during
        validation. See `tforce_val` or `teacher_force_val` to use
        teacher forcing during validation.
    "tforce_val": bool
        if true, the correct language inputs are fed into the model
        during validation. Only implemented for ENGLISH language, no
        `actnlish`
    "lang_teacher_p": float [0,1]
        the probability of using teacher forcing on the language inputs
        for each training iteration. only applies if incl_lang_inpt
        is true.
    "shuffle_teacher_lang": bool
        if true, will shuffle the teacher forced lang intputs during
        training. Does not do so during validation.
    "teacher_force_val": bool
        if true, the correct language inputs are fed into the model
        during validation. Only implemented for ENGLISH language, no
        `actnlish`

    "lang_inpt_drop_p": float [0,1]
        the dropout probability on the embeddings of the lang inputs.
        only applies if incl_lang_inpt is true
    "bottleneck": bool
        if true, with the NSepLSTM and SeparateLSTM, the input to the
        action lstm(s) is only the language prediction (without the
        visual latent vector). If `actnlish`
        is false, the game will always spawn the agent in the same
        position relative to the pile and ending button. Otherwise the
        game is unsolvable.

    "output_fxn": str
        the name of the output function for the model. if no output
        fxn is wanted, leave as null or specify "NullOp"
    "h_size": int
        this number is used as the size of the RNN hidden vector and
        the transformer dim
    "emb_ffn": bool
        if true, the embedding is processed through a feedforward network.
    "inpt_consol_emb_size": int
        if using word embeddings as language input, can specify
        the dimensionality of the embeddings
    "learn_h": bool
        determines if the hidden vectors should be learned or not. if
        true, both the h and c vectors are intiialized to a learned
        vector.
    "h_mult": int
        this number is a multiplier for the `h_size` to expand the
        dimensionality of the dense output hidden layers.
    "n_outlayers": int
        the number of layers in the action and language output networks
        only applies when using Vary model variants
    "bnorm": bool
        determines if the model should use batchnorm. true means it
        does use batchnorm
    "legacy": bool
        determines if the fc nets will take on a legacy architecture
    "lnorm": bool
        determines if the model should use layernorm. true means it
        does use layernorm on both the h and c recurrent vectors just
        after the lstm cell. This is overriden in cases where `c_lnorm`
        is false. Note that using the layernorm after the
        cell still results in a normalized input for the next step
        in time while normalizing the input for the action and language
        networks.
    "c_lnorm": bool
        determines whether or not lnorm should be performed on the
        c vector. You probably want this to be false.
    "lang_lnorm": bool
        if true, will add an lnorm before the language lstms
    "fc_lnorm": bool
        if true, the model uses a layernorm before each Linear layer
        in the fully connected layers
    "fc_bnorm": bool
        if true, the model uses a batchnorm before each Linear layer
        in the fully connected layers
    "scaleshift": bool
        if true, adds a ScaleShift layer after each Linear layer in
        the fully connected layers
    "skip_lstm": bool
        if true, the features are inluded using a skip connection
        to the second lstm. Only applies in DoubleVaryLSTM variants
    "actv_fxn": str
        the activation function for the output layers. defaults to "ReLU"
    "n_frame_stack": int
        the number of frames to stack for an observation of the game
    "lr": float
        the learning rate
    "l2": float
        the weight decay or l2 regularization parameter
    "lang_p": float between 0 and 1
        the portion of the loss during the second phase attributed
        to language.
    "null_alpha": float
        a hyperparameter to adjust how much weight should be placed
        on producing zeros for the base numeral system outputs
        following the STOP token. loss is calculated as
        loss += null_alpha*null_loss
        (It is not a proportionality parameter like `lang_p`)
    "conv_noise": float
        the standard deviation of gaussian noise applied to the
        convolutional layers of the model. if 0, has no effect
    "dense_noise": float
        the standard deviation of gaussian noise applied to the
        dense layers of the model. if 0, has no effect
    "feat_drop_p": float
        the probability of zeroing a neuron within the features
        of the cnn output.
    "drop_p": float
        the probability of zeroing a neuron within the dense
        layers of the network.

    "depths": tuple of ints
        the depths of the cnn layers. the number of cnn layers is
        dependent on the number of items in this tuple. This is
        also used to determine the embedding size of vision transformers
    "kernels": tuple of ints
        the kernel sizes of the cnn layers. the number of items in this
        tuple should match that of the number of items in `depths`
    "strides": tuple of ints
        the strides of the cnn layers. the number of items in this
        tuple should match that of the number of items in `depths`
    "paddings": tuple of ints
        the paddings of the cnn layers. the number of items in this
        tuple should match that of the number of items in `depths`

    "pre_grab_count": bool
        if true, the language system will be trained to predict the
        number of response items that will result from interacting
        with the dispenser. Defaults to false which trains the lang
        system to predict the number of response items currently on
        the grid.
    "langall": bool
        if false, language predictions only occur when the agent drops
        an object. Otherwise the language predictions occur at every
        step. defaults to false. this parameter overrides `count_targs`
        and `lang_targs_only`. overridden by actnlish
    "count_targs": bool
        LARGELY DEPRECATED!!! automatically set to false when
        `use_count_words` is 0, otherwise set to true.

        Only applies to v4, v7, and v8 variants of gordongames. if true,
        the model will learn to count out the targets in addition to
        the items. If false, model will only count the items. Differs
        from langall=True in that it skips the steps that
        the agent takes to move about the grid. Overridden by
        `lang_targs_only`. Forced to be False
    "lang_targs_only": int
        Only applies to v4, v7, and v8 variants of gordongames. if 0,
        effectively does nothing. If 1, the language labels will only
        be for the targets. No counting is performed on the items.
        This argument is overridden by langall being true.
        count_targs is overridden by this argument. drop_perc_threshold
        has no impact on this argument.

    "env_types": list of str
        the name of the gym environments to be used for the training.
        each environment spawns a new data collection process in the
        training. If you want to use only one environment with multiple
        data collection processes, specify the same environment
        repeatedly. `batch_size` will be forced to be divisible by the
        length of this list.
    "incl_cdtnl": bool
        if true, guarantees conditional vector is used in model.
        If false, the conditional vector is only used when there are
        more than 1 environment types
    "harsh": bool
        an optional parameter to determine the reward scheme for
        gordongames variants
    "pixel_density": int
        the side length (in pixels) of a unit square in the game. Only
        applies to gordongames variants
    "grid_size": list of ints (height, width)
        the number of units (height, width) of the game. Only applies
        to gordongames variants
    "min_play_area": bool
        if true, minimizes the play area (area above the
        dividing line of the grid) to 4 rows. Otherwise,
        dividing line is placed at approximately the middle
        row of the grid.
    "rand_pdb": bool
        if true, the player, dispenser, and ending button are randomly
        placed along the top row of the grid at the beginning of each
        episode. If false, they are evenly spaced along the top row
        in the following order player, dispenser, ending button.
    "spacing_limit": int or null
        if not null and greater than 0, limits the spacing between the
        player and dispenser, and the ending button and dispenser to
        be within `spacing_limit` steps on either side of the
        dispenser's initial position. If `rand_pdb` is false, the
        player and ending button will always be `spacing_limit` steps
        away symmetrically centered on the dispenser.
    "sym_distr": bool
        if false and `rand_pdb` is false, the player, dispenser, and
        button are consistently distributed the same way at
        initialization on every episode. Otherwise the initial
        distribution is reflected about the yaxis with 50% prob. Only
        applies when `rand_pdb` is false.
    "rand_pdb": bool
        if true, the player, dispenser (aka pile), and ending button
        are randomly distributed along the top row at the beginning of
        the game. Otherwise they are deterministically set.
    "player_on_pile": bool
        if true, the player always starts on top of the dispenser pile
        in counting games. If false, it will not.
    "spacing_limit": int
        if greater than 0, limits the spacing between the player and
        dispenser, and the ending button and dispenser to be within
        `spacing_limit` steps on either side of the dispenser's initial
        position. If `rand_locs` is false, the player and ending
        button will always be `spacing_limit` steps away symmetrically
        centered on the dispenser.
    "rand_timing": bool
        if true, the timing of the initial display phase is stochastic
        so that the agent cannot simply count the number of frames
        rather than the number of target items.
    "timing_p": float between 0 and 1
        the probability of an animation step displaying the next target
        object. A value of 1 means the agent could count the number of
        frames instead of the number of target items. A value of 0 will
        not allow the game to progress past the animation phase.
    "n_held_outs": int or None
        the number of held out locations for each target quantity in
        the gordongames
    "center_signal": bool
        determines where the response phase signal pixel goes. if true,
        it will be centered in the demonstration half of the grid.
        if false, it will go in the rightmost column one down from the
        topmost row.
    "targ_range": list of ints (low, high)
        the range of potential target counts. This acts as the default
        if lang_range or actn_range are not specified. both low and
        high are inclusive. only applies to gordongames variants.
    "lang_range": list of ints (low, high)
        the range of potential target counts for training the language
        model during phase 0. both low and high are
        inclusive. only applies to gordongames variants.
    "actn_range": list of ints (low, high)
        the range of potential target counts for training the action
        model during phases other than 0. both low and high are
        inclusive. only applies to gordongames variants.

    "hold_outs": set (or list) of ints
        the targets/item counts to hold out during training. influences
        `hold_lang` and `hold_actns` depending on their respective
        values. See the documentation below for more detail
    "hold_lang": set (or list) of ints or bool
        the language targets to hold out during training. if true,
        defaults to `hold_outs`. if None or False, no langauge is held
        out. if a set or list, then the contained integers represent
        the `n_items` values that are held out from the language training
    "hold_actns": set (or list) of ints or bool
        the ending game target quantities to hold out during training.
        if true, defaults to value of `hold_outs`. if False or None, no
        target quantities are held out. if a set or list, then the game
        never has an ending target quantity of the listed integers.
        Important to note that the held target quantities are still
        contained within episodes with target quantities greater than
        the held target quantity. i.e. holding a target quanity of 6
        will not hold the necessary 6 value from training on target
        quantities of 7 or greater.

    "log_samp": bool
        if true, will sample from a log distribution instead of a
        zifpian distribution. The log sampling uses `zipf_order` for
        the following proportional probability where max is the max
        target value and k is the sampled target value.
            `p(k) ~ (-log(k) + log(max) + 1)^zipf_order`
    "zipf_order": float or None
        if greater than 0, the targets are drawn proportionally to the
        zipfian distribution using `zipfian_order` as the exponent.

    "batch_size": int
        the number of rollouts to collect when collecting data and
        the batch size for the training. This parameter will be changed
        so that it is divisible by the number of environments listed
        in `env_types`.
    "seq_len": int
        the number of consecutive frames to feed into the model for
        a single batch of data
    "min_seq_len": int
        the seq length to start from if using `incr_seq_len` and the
        minimum possible `seq_len` if using rand_seq_len. Defaults to 7.
    "incr_seq_len": bool
        if true, the seq_len for bptt will start small and increment
        upwards over the course of the whole training.
    "rand_seq_len": bool
        if true and roll_data is true, the sequence length is sampled
        uniformly from 7 to the argued `seq_len`.
    "exp_len": int
        the "experience" length for a single rollout. This is the
        number of steps to take in the environment for a single row
        in the batch during data collection.
    "max_steps": int or None
        optional argument, if argued, determines the maximum number of
        steps that an episode can take. Be careful to make completion
        of an episode possible within the argued number of steps!
    "randomize_order": bool
        if true, the training will randomize the order of the sequence
        chunks. This means that the elements of the sequence will not
        be randomized, but the order of the sequences will. If false,
        the sequences maintain the order they were collected in. If
        your model is stateful (i.e. recurrent), you probably want this
        to be false. Otherwise, true is probably good (for non-recurrent
        transformer types).

    "n_heads": int
        the number of transformer attention heads
    "n_vit_layers": int
        the number of vision transformer encoder layers
    "n_layers": int
        the number of transformer encoder layers
    "max_ctx_len": null or int
        if you would like to increase the context length for trainings
        using the transformer class without increasing the back-
        propagation context-window, you can set this value higher than
        `seq_len`. This will make the context window larger without
        backpropping through the increased number of tokens.
    "stack_context": bool
        if true, transformer tokens will consist of the concatenation
        of the visual latent vector with the language embedding. These
        will each be half of h_size, combined to make h_size tokens for
        the transformer context. Otherwise the language and vision
        vectors will alternate in context. Their ordering in context
        will depend on lstm_lang_first.

    "n_inner_loops": 1
        the number of times to train on one collected iteration of data.
        this effectively repeats each epoch n times. Makes most sense
        to use when shuffling data order.
    "reset_trn_env": bool
        if true, the training environments are reset at the beginning
        of each collection. This ensures that the model never has to
        jump into the middle of an episode during training.
    "roll_data": bool
        if true, training data is fed in such that the next training
        loop is only one time step in the future from the first step
        in the last loop (so seq_len-1 steps overlap with the previous
        loop). Otherwise training data is fed in such that
        the first data point in the current training loop is 1 time
        step after the last time step in the last loop (so there is
        no data overlap).

    "val_targ_range": list of ints (low, high) or None
        the range of target counts during the validation phase. both
        low and high are inclusive. only applies to gordongames
        variants. if None, defaults to training range.
    "val_max_actn": bool (deprecated: use `val_temp` instead)
        if true, actions during the validation phase are selected as
        the maximum argument over the model's action probability output.
        If False, actions are sampled from the action output with
        probability equal to the corresponding output probability
    "val_temp": float or None
        the action sampling temperature for validation rollouts. if
        none or 0, will use argmax
    "val_mod": int or null
        a modulus to determine which epochs to validate. This speeds
        up trainings. if None or 0, will default to validating every
        epoch
    "doubl_val_mod": null or list of ints
        a list of epochs at which `val_mod` should be doubled from its
        current value. This allows validation every other epoch for
        the first x epochs, and then every 4th for the next y epochs,
        and so on.
    "always_epochs": null or int
        a threshold epoch before which all epochs are validated
        regardless of `val_mod`.
    "n_eval_eps": int or null
        the number of episodes to collect for each target value during
        validation.

    "oracle_type": str
        the name of the class to use for the oracle. i.e. "GordonOracle"
    "drop_grad": float (between 0 and 1)
        the probability of dropping gradient values. higher values means
        more dropout. This is applied to all model parameters after
        every backward pass before the optimizer update step.
    "grad_norm": float
        the maximum gradient norm used for gradient norm clipping. If
        the argued value is less than or equal to 0, no gradient clipping
        occurs.
    "optim_type": str
        the name of the class to use for the optimizer. i.e. "Adam"
    "factor": float
        the factor to decrease the learning rate by. new_lr = factor*lr
    "patience": int
        the number of epochs to wait for loss improvement before
        reducing the learing rate
    "threshold": float
        the threshold by which to determine if the training has
        plateaued. smaller threshold means smaller changes count as
        meaningful progress in the training. This is proportional to
        the size of the loss.
        loss_to_beat = best_actual_loss*(1-threshold)
        If loss_to_beat is further away from actual loss, then a
        learning rate change is more likely to occur.
    "min_lr": float
        the minimum learning rate allowed by the scheduler
    "preprocessor": str
        the name of the preprocessing function. this function operates
        directly on observations from the environment before they are
        concatenated to form the "state" of the environment
    "best_by_key": str
        the name of the metric to use for determining the best
        performing model. i.e. "val_perc_correct_avg"

## Notes on Tranformer Model Type Trainings
In general you will want the following hyperparameter settings:

    "randomize_order": true,
    "roll_data": false,
    "exp_len": 5000, # just use a longer exp_len here than for LSTMs
    "seq_len": 96, # Use a longer seq_len here than for LSTMs
    "rand_seq_len": false,

Some of the above settings will automatically occur, but it is best
to be explicit where you can.

## Note on Pre Navigation Trainings
The easiest way to do a pre\_nav training is to first train on the
navigation task and then to perform a standard training using the
pre\_nav models as the starting checkpoint.
