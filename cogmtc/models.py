import collections
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import *
from cogmtc.utils.torch_modules import *
from cogmtc.utils.utils import update_shape, get_transformer_fwd_mask, max_one_hot, INEQUALITY, ENGLISH, PIRAHA, RANDOM, DUPLICATES, NUMERAL

from cogmtc.envs import TORCH_CONDITIONALS, CDTNL_LANG_SIZE
import matplotlib.pyplot as plt

class MODEL_TYPES:
    LSTM = "LSTM"
    TRANSFORMER = "TRANSFORMER"
    CNN = "CNN"
    ALL_TYPES = {LSTM, TRANSFORMER, CNN}
    
    @staticmethod
    def GETTYPE(string):
        if "Transformer" in string:
            return MODEL_TYPES.TRANSFORMER
        elif "Cnn" in string:
            return MODEL_TYPES.CNN
        else: return MODEL_TYPES.LSTM


class LANGACTN_TYPES:
    SOFTMAX = 0
    ONEHOT = 1
    HVECTOR = 2
    CONSOLIDATE = 3

def get_fcnet(inpt_size,
              outp_size,
              n_layers=2,
              h_size=256,
              noise=0,
              drop_p=0,
              bnorm=False,
              lnorm=False,
              scaleshift=True,
              legacy=False,
              actv_fxn="ReLU"):
    """
    Defines a simple fully connected Sequential module

    Args:
        inpt_size: int
            the dimension of the inputs
        outp_size: int
            the dimension of the final output
        n_layers: int
            the number of layers for the fc net
        h_size: int
            the dimensionality of the hidden layers
        noise: float
            the std of added noise before the relue at each layer.
        drop_p: float
            the probability of dropping a node
        bnorm: bool
            if true, batchnorm is included before each relu layer
        lnorm: bool
            if true, layer norm is included before each relu layer
        scaleshift: bool
            if true, a ScaleShift layer is added after the activation
            function
        legacy: bool
            if true, matches architecture of legacy models
    """
    outsize= h_size if n_layers > 1 else outp_size
    block = [  ]
    block.append( nn.Linear(inpt_size, outsize) )
    prev_size = outsize
    for i in range(1, n_layers):
        block.append( GaussianNoise(noise) )
        if legacy and lnorm: block.append( nn.LayerNorm(outsize) )
        block.append( nn.Dropout(drop_p) )
        block.append( globals()[actv_fxn]() )
        if bnorm: block.append( nn.BatchNorm1d(outsize) )
        if not legacy and lnorm: block.append( nn.LayerNorm(outsize) )
        if scaleshift: block.append( ScaleShift((outsize,)) )
        if i+1 == n_layers: outsize = outp_size
        block.append( nn.Linear(prev_size, outsize) )
    return nn.Sequential(*block)

class CoreModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def is_cuda(self):
        try:
            return next(self.parameters()).is_cuda
        except:
            return False

    def get_device(self):
        try:
            d = next(self.parameters()).get_device()
            if d < 0: return "cpu"
            return d
        except:
            return "cpu"

class Model(CoreModule):
    """
    This is the base class for all models within this project. It
    ensures the appropriate members are added to the model.

    All models that inherit from Model must implement a step function
    that takes a float tensor of dims (B, C, H, W)
    """
    def __init__(self,
        inpt_shape,
        actn_size,
        lang_size,
        n_lang_denses=1,
        h_size=128,
        h_mult=2,
        bnorm=False,
        lnorm=False,
        conv_noise=0,
        dense_noise=0,
        feat_drop_p=0,
        drop_p=0,
        lang_inpt_drop_p=0,
        lstm_lang_first=True,
        env_types=["gordongames-v4"],
        n_heads=8,
        n_layers=3,
        n_vit_layers=3,
        n_outlayers=2,
        seq_len=64,
        output_fxn="NullOp",
        actv_fxn="ReLU",
        depths=[32, 48],
        kernels=[3, 3],
        strides=[1, 1],
        paddings=[0, 0],
        skip_lstm=False,
        max_char_seq=1,
        STOP=1,
        null_idx=0,
        lstm_lang=False,
        incl_lang_inpt=True,
        incl_actn_inpt=False,
        langactn_inpt_type=LANGACTN_TYPES.SOFTMAX,
        zero_after_stop=False,
        use_count_words=None,
        max_ctx_len=None,
        vision_type=None,
        learn_h=False,
        scaleshift=True,
        fc_lnorm=False,
        c_lnorm=True,
        lang_lnorm=False,
        fc_bnorm=False,
        stagger_preds=True,
        bottleneck=False,
        extra_lang_pred=False,
        legacy=False,
        *args, **kwargs
    ):
        """
        Args: 
            inpt_shape: tuple or listlike (..., C, H, W)
                the shape of the input
            actn_size: int
                the number of potential actions
            lang_size: int
                the number of potential words
            n_lang_denses: int
                the number of duplicate language model outputs
            h_size: int
                this number is used as the size of the RNN hidden 
                vector and the transformer dim
            h_mult: int
                this number is a multiplier for the `h_size` to expand
                the dimensionality of the dense output hidden layers.
            bnorm: bool
                if true, the model uses batch normalization
            lnorm: bool
                if true, the model uses layer normalization on the h
                and c recurrent vectors after the recurrent cell
            conv_noise: float
                the standard deviation of noise added after each
                convolutional operation
            dense_noise: float
                the standard deviation of noise added after each
                dense operation (except for the output)
            feat_drop_p: float
                the probability of zeroing a neuron within the features
                of the cnn output.
            drop_p: float
                the probability of zeroing a neuron within the dense
                layers of the network.
            lang_inpt_drop_p: float
                the dropout probability on the embeddings of the lang
                inputs
            lstm_lang_first: bool
                only used in multi-lstm model types. If true, the h
                vector from the first LSTM will be used as the input
                to the language layers. The second h vector will then
                be used for the action layers. If False, the second h
                vector will be used for language and the first h for
                actions.
            env_types: list of str
                a list of environment types
            n_heads: int
                the number of attention heads if using a transformer
            n_layers: int
                the number of transformer layers.
            n_vit_layers: int
                the number of vision transformer layers.
            n_outlayers: int
                the number of layers for the actn and language dense
                network output modules if using the Vary variants of
                the LSTM models
            seq_len: int
                an upper bound on the sequence length
            output_fxn: str
                the string is converted into the respective torch module.
                this module operates on the action outputs of each
                model. it must take a tensor and return a tensor of the
                same shape. Assume inputs are of dimenion (B, ..., E).
                make sure the module has not yet been instantiated
            actv_fxn: str
                the name of the activation function for each layer
                of the action and language outputs.
            depths: tuple of ints
                the depth of each layer of the fully connected output
                networks
            kernels: tuple of ints
                the kernel size of each layer of the fully connected
                ouput networks
            strides: tuple of ints
                the stride of each layer of the fully connected
                ouput networks
            paddings: tuple of ints
                the padding of each layer of the fully connected
                ouput networks
            skip_lstm: bool
                if true, the features are inluded using a skip connection
                to the second lstm. Only applies in DoubleLSTM variants
            max_char_seq: int or None
                if int, it is the number of language tokens to predict
                at every step
            STOP: int
                the index of the STOP token (if one exists). Only
                necessary for NUMERAL type models.
            null_idx: int
                the index of the NULL token (if one exists). Only
                necessary for NUMERAL type models.
            lstm_lang: bool
                if you want to use an additional lstm to output the
                language for numeral systems, set this to true. if false
                and using a numeral system, a single dense net makes
                all numeral predictions at the same time. Does not
                affect anything if not using numeral system
            incl_actn_inpt: bool
                if true, for the SymmetricLSTM, the softmax or one-hot
                encoding or h vector (depending on `langactn_inpt_type`)
                of the action output for the last timestep is included as
                input into the language and action lstms. If false,
                the action output is not included.
            incl_lang_inpt: bool
                if true, for the SymmetricLSTM, the softmax or one-hot
                encoding or h vector (depending on `langactn_inpt_type`)
                of the language output for the last timestep is included
                as input into the language and action lstms. If false,
                the language output is not included.
            langactn_inpt_type: int
                Pretains to the incl_actn_inpt and incl_lang_inpt.
                Determines whether the input should be the softmax of
                the output, a one-hot encoding of the output, or the
                recurrent state vector that produced the output.

                options are:
                    0: LANGACTN_TYPES.SOFTMAX
                    1: LANGACTN_TYPES.ONEHOT
                    2: LANGACTN_TYPES.HVECTOR
                    3: LANGACTN_TYPES.CONSOLIDATE
            zero_after_stop: bool
                only used for the NUMERAL trainings when
                `langactn_inpt_type` is equal to 0 or 1 and
                `incl_lang_inpt` is true. If `zero_after_stop` is true,
                all values following a STOP prediction (including the
                STOP prediction itself) in the language prediction that
                is used as input on the next time step are set to zero.
            use_count_words: int
                the type of language training
            max_ctx_len: int or None
                the maximum context length for the transformer models
                (not including the ViT)
            vision_type: str
                the model class to be used for the raw visual processing
                options include (but are probably not limited to):
                VaryCNN, ViT
            learn_h: bool
                if true, the recurrent vectors are learned
            scaleshift: bool
                if true, adds a scaleshift layer after each Linear
                layer in the fully connected layers
            fc_lnorm: bool
                if true, adds a layernorm layer before each Linear
                layer in the fully connected layers
            c_lnorm: bool
                if true, performs layernorm on all relevant c vectors
            lang_lnorm: bool
                if true, adds a layernorm layer before the language
                lstm in the SeparateLSTM model
            fc_bnorm: bool
                if true, adds a batchnorm layer before each Linear
                layer in the fully connected layers
            stagger_preds: bool
                if true, the language and action predictions are made
                from different LSTMs within the model if possible. The
                order is determined using `lstm_lang_first`
            same_step_lang (deprecated): bool
                ~~This argument has been deprecated. All models that
                use lang inputs use the equivalent of `same_step_lang`
                equal to true.~~
                If true, the language prediction used as input to the
                action network comes from the same time step. i.e. the
                language prediction is made and directly feeds into the
                action prediction.  Only applies if `incl_lang_inpt` is
                true.
            bottleneck: bool
                if true, only the language predictions are fed into
                the action module. i.e. no vision is fed into the action
                module. Otherwise, both lang and vision are fed into
                actn module. Only applies if using incl_lang_inpt
            extra_lang_pred: bool
                if true, the DoubleVaryLSTM will include an extra
                language prediction system that it will use for making
                language predictions in the incl_lang_inpt cases
                (emulating the behavior of the SeparateLSTM). This is
                in addition to the DoubleVaryLSTM's usual
                language predictions. Only relevant when using the
                DoubleVaryLSTM model type and incl_lang_preds is true.

                Also relevant if using the DblBtlComboLSTM. If this
                is true, the model makes an extra language prediction
                from the hidden state of the action lstm. Otherwise
                the action lstm is unadulterated.
            legacy: bool
                if true, the fc nets use a legacy architecture
        """
        super().__init__()
        self.model_type = MODEL_TYPES.LSTM
        self.inpt_shape = inpt_shape
        self.actn_size = actn_size
        self.lang_size = lang_size
        self.h_size = h_size
        self.h_mult = h_mult
        self.bnorm = bnorm
        self.lnorm = lnorm
        self.conv_noise = conv_noise
        self.dense_noise = dense_noise
        self.feat_drop_p = feat_drop_p
        self.drop_p = drop_p
        self.lang_inpt_drop_p = lang_inpt_drop_p
        self.n_lang_denses = n_lang_denses
        self._trn_whls = nn.Parameter(torch.ones(1), requires_grad=False)
        self.lstm_lang_first = lstm_lang_first
        self.n_lstms = 1
        self.env_types = env_types
        self.env2idx = {k:i for i,k in enumerate(self.env_types)}
        self.n_envs = len(self.env_types)
        self.initialize_conditional_variables()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_vit_layers = n_vit_layers
        self.seq_len = seq_len
        self.max_char_seq = max_char_seq
        self.STOP = STOP
        self.null_idx = null_idx
        self.lstm_lang = lstm_lang
        if output_fxn is None: output_fxn = "NullOp"
        self.output_fxn = globals()[output_fxn]()
        self.actv_fxn = actv_fxn
        self.n_outlayers = n_outlayers
        self.depths = [self.inpt_shape[-3], *depths]
        self.kernels = kernels
        if isinstance(kernels, int):
            self.kernels=[kernels for i in range(len(depths))]
        self.strides = strides
        if isinstance(strides, int):
            self.strides=[strides for i in range(len(depths))]
        self.paddings = paddings
        if isinstance(paddings, int):
            self.paddings=[paddings for i in range(len(depths))]
        self.skip_lstm = skip_lstm
        self.incl_actn_inpt = incl_actn_inpt
        self.incl_lang_inpt = incl_lang_inpt
        self.langactn_inpt_type = langactn_inpt_type
        self.zero_after_stop = zero_after_stop
        self.use_count_words = use_count_words
        self.max_ctx_len = 128 if max_ctx_len is None else max_ctx_len
        self.vision_type = vision_type
        self.learn_h = learn_h
        self.scaleshift = scaleshift
        self.fc_lnorm = fc_lnorm
        self.c_lnorm = c_lnorm
        self.lang_lnorm = lang_lnorm
        self.fc_bnorm = fc_bnorm
        self.stagger_preds = stagger_preds
        self.bottleneck = bottleneck
        self.extra_lang_pred = extra_lang_pred
        self.legacy = legacy

    def initialize_conditional_variables(self):
        """
        Creates the conditional lstm, the conditional long indices, and
        the conditional batch distribution tensor. The cdtnl_batch
        tensor is a way to use the same conditional for all appropriate
        batch rows at the same time. At training time, we use 
        `repeat_interleave` to expand cdtnl_batch appropriately.
        """
        self.cdtnl_lstm = ConditionalLSTM(
            self.h_size, lang_size=CDTNL_LANG_SIZE
        )
        max_len = max([len(v) for v in TORCH_CONDITIONALS.values()])
        cdtnl_idxs = torch.zeros(len(self.env_types),max_len).long()
        for env_type in self.env_types:
            k = self.env2idx[env_type]
            l = len(TORCH_CONDITIONALS[env_type])
            cdtnl_idxs[k,:l] = TORCH_CONDITIONALS[env_type]
        self.register_buffer("cdtnl_idxs", cdtnl_idxs)

    def make_actn_dense(self, inpt_size=None):
        if inpt_size==None: inpt_size = self.h_size
        self.actn_dense = get_fcnet(
            inpt_size,
            self.actn_size,
            n_layers=self.n_outlayers,
            h_size=self.h_size*self.h_mult,
            noise=self.dense_noise,
            drop_p=self.drop_p,
            actv_fxn=self.actv_fxn,
            bnorm=self.fc_bnorm,
            lnorm=self.fc_lnorm,
            scaleshift=self.scaleshift,
            legacy=self.legacy
        )

    def make_lang_denses(self, inpt_size=None):
        if inpt_size==None: inpt_size = self.h_size
        self.lang_denses = nn.ModuleList([])
        # In case we're actually making multiple language predictions
        # from a single output
        lang_size = self.lang_size
        if self.max_char_seq is not None and self.max_char_seq>1:
            # if lstm_lang is true, the language lstm needs the base
            # numeral as its language size. if false, we can think of
            # the output as a concatenated vector
            if not self.lstm_lang:lang_size=lang_size*self.max_char_seq
        for i in range(self.n_lang_denses):
            if self.lstm_lang:
                dense = NumeralLangLSTM(
                    inpt_size=inpt_size,
                    h_size=self.h_size,
                    lang_size=lang_size,
                    max_char_seq=self.max_char_seq,
                    n_outlayers=self.n_outlayers,
                    h_mult=self.h_mult,
                    drop_p=self.drop_p,
                    actv_fxn=self.actv_fxn,
                    lnorm=self.lnorm,
                    fc_lnorm=self.fc_lnorm,
                    fc_bnorm=self.fc_bnorm,
                    scaleshift=self.scaleshift
                )
            else:
                dense = get_fcnet(
                    inpt_size=inpt_size,
                    outp_size=lang_size,
                    n_layers=self.n_outlayers,
                    h_size=self.h_size*self.h_mult,
                    noise=self.dense_noise,
                    drop_p=self.drop_p,
                    actv_fxn=self.actv_fxn,
                    bnorm=self.fc_bnorm,
                    lnorm=self.fc_lnorm,
                    scaleshift=self.scaleshift,
                    legacy=self.legacy
                )
            self.lang_denses.append(dense)

    @property
    def trn_whls(self):
        """
        This is essentially a boolean used to communicate if the
        runners should use the model predictions or the oracle
        predictions for training.

        Returns:
            training_wheel_status: int
                if 1, then the training wheels are still on the bike.
                if 0, then that sucker is free to shred
        """
        return self._trn_whls.data[0].item()

    @trn_whls.setter
    def trn_whls(self, status):
        """
        Sets the status of the training wheels

        Args:
            status: int
                if set to 1, the training wheels are considered on and
                the data collection will use the oracle. if 0, then
                the training data collection will use the actions from
                the model but still use the oracle's actions as the
                labels.
        """
        self._trn_whls.data[0] = status

    def reset(self, batch_size):
        """
        Only necessary to override if building a recurrent network.
        This function should reset any recurrent state in a model.

        Args:
            batch_size: int
                the size of the incoming batches
        """
        pass

    def reset_to_step(self, step=1):
        """
        Only necessary to override if building a recurrent network.
        This function resets all recurrent states in a model to the
        recurrent state that occurred after the first step in the last
        call to forward.

        Args:
            step: int
                the index of the step to revert the recurrence to
        """
        pass

    def step(self, x):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            actn: torch Float Tensor (B, K)
            lang: torch Float Tensor (B, L)
        """
        pass

    def forward(self, x):
        """
        Performs multiple steps in time rather than a single step.

        Args:
            x: torch FloatTensor (B, S, C, H, W)
        Returns:
            actn: torch Float Tensor (B, S, K)
            lang: torch Float Tensor (B, S, L)
        """
        pass

class NumeralLangLSTM(nn.Module):
    """
    This is a module to assist in producing numerals recurrently within
    the current project structure.
    """
    def __init__(self, inpt_size,
                       h_size,
                       lang_size,
                       max_char_seq=4,
                       n_outlayers=1,
                       h_mult=3,
                       drop_p=0,
                       actv_fxn="ReLU",
                       lnorm=True,
                       fc_lnorm=False,
                       fc_bnorm=False,
                       scaleshift=False,
                       legacy=False,
                       *args,**kwargs):
        """
        Args:
            inpt_size: int
            h_size: int
            lang_size: int
            max_char_seq: int
            n_outlayers: int
            drop_p: float
            lnorm: bool
            fc_lnorm: bool
            fc_bnorm: bool
            scaleshift: bool
        """
        super().__init__()
        self.inpt_size = inpt_size
        self.h_size = h_size
        self.lang_size = lang_size
        self.n_loops = max_char_seq
        self.n_outlayers = n_outlayers
        self.h_mult = h_mult
        self.lnorm = lnorm
        self.fc_lnorm = fc_lnorm
        self.fc_bnorm = fc_bnorm
        self.scaleshift = scaleshift
        self.drop_p = drop_p
        self.actv_fxn = actv_fxn
        self.legacy = legacy
        self.lstm = GenerativeLSTM(
            self.h_size,self.h_size,lnorm=self.lnorm
        )
        self.dense = get_fcnet(
            inpt_size=self.h_size,
            outp_size=self.lang_size,
            n_layers=self.n_outlayers,
            h_size=self.h_size*self.h_mult,
            noise=0,
            drop_p=self.drop_p,
            actv_fxn=self.actv_fxn,
            lnorm=self.fc_lnorm,
            bnorm=self.fc_bnorm,
            scaleshift=self.scaleshift,
            legacy=self.legacy
        )

    def forward(self, x):
        """
        Args:
            x: torch tensor (B, H)
        Returns:
            fx: torch tensor (B, N*lang_size)
        """
        fx = self.lstm(x, self.n_loops)
        B,N,H = fx.shape
        fx = self.dense(fx.reshape(-1,H))
        return fx.reshape(B,-1)

def identity(x, *args, **kwargs):
    return x

class InptConsolidationModule(nn.Module):
    """
    This is a module to assist in converting the raw output from a
    language or action prediction into a single vector representation
    to be used as input to an LSTM module in the next timestep.

    """
    def __init__(self, lang_size,
                       use_count_words=None,
                       h_size=None,
                       max_char_seq=1,
                       STOP=1,
                       null_idx=0,
                       drop_p=0,
                       *args,**kwargs):
        """
        Args:
            lang_size: int
                size of the inputs that need a transformation or
                consolidation
            use_count_words: int
                the type of language training
            h_size: int
                hidden size of lstm if consolidating a sequence
            max_char_seq: int
            STOP: int
                index of stop token (if one exists). only matters if
                max_char_seq is greater than 1
            null_idx: int
                index of NULL token (if one exists). only matters if
                max_char_seq is greater than 1
            drop_p: float
        """
        super().__init__()
        self.lang_size = lang_size
        self.h_size = h_size
        self.use_count_words = use_count_words
        self.mcs = 1 if max_char_seq is None or max_char_seq < 1\
                     else max_char_seq
        self.STOP = STOP
        self.null_idx = null_idx
        self.drop_p = drop_p

        self.embeddings = nn.Embedding(self.lang_size,self.h_size)
        self.dropout = nn.Dropout(p=self.drop_p)
        if self.use_count_words == NUMERAL:
            self.lstm_consol = ContainedLSTM( self.h_size, self.h_size )
        self.consolidator = nn.Sequential(
            nn.Linear(self.h_size, self.h_size),
            nn.LayerNorm(self.h_size),
            nn.ReLU()
        )
        self.proj = nn.Linear(self.h_size, self.h_size)

    def reshape_and_extract(self, inpt, *args, **kwargs):
        """
        Only used for variable length number sequences. This returns
        a function that reshapes the input to the mcs, performs a
        softmax (or equivalent) and then reshapes back.

        inpt: torch Tensor (..., N)
            N must be divisible by self.mcs
        """
        if self.langactn_inpt_type == LANGACTN_TYPES.SOFTMAX:
            fxn = nn.functional.softmax
        else:
            fxn = max_one_hot
        og_shape = inpt.shape
        inpt = inpt.reshape((len(inpt), self.mcs, -1))
        # Fail safe so that a stop prediction always exists at last
        # entry in the sequence
        inpt[:, -1, self.STOP] = inpt[:,-1].max(-1)[0]+1
        inpt = fxn( inpt.float(), dim=-1 ).reshape(og_shape)
        return inpt

    def get_mask(self, inpt, token, incl_token=False):
        """
        This returns a boolean mask for all indices following (and
        potentially including) the first index that is equal to the
        argued token in the inpt sequence.

        inpt: torch tensor (B, S, N)
            N must be divisible by self.mcs
        token: int
        incl_token: bool
            if true, the mask includes the first occurence of the
            argued token.
        """
        mask = (inpt==token).bool()
        if not incl_token:
            mask = mask.roll(1,dims=-1)
            mask[:,0] = False
        for i in range(1, self.mcs): # need loop for cumulative value
            mask[:,i] = mask[:,i-1]|mask[:,i]
        return mask

    def set_zero_after_stop(self, inpt, *args, **kwargs):
        """
        Only used for variable length number sequences. This returns
        a function that locates the STOP token prediction and zeros
        all following predictions

        inpt: torch tensor (B, N)
            N must be divisible by self.mcs
        """
        og_shape = inpt.shape
        if self.mcs is None or self.mcs <= 1: return inpt
        # inpt at this point is shaped (B, N) but we want (B, M, N)
        inpt = inpt.reshape((len(inpt), self.mcs, -1))
        argmaxes = torch.argmax(inpt, dim=-1) # (B,M)
        mask = (argmaxes==self.STOP).float()
        for i in range(1, self.mcs):
            mask[:,i] = mask[:,i-1]+mask[:,i]
        mask[mask>1] = 1
        inpt = inpt*(1-mask.unsqueeze(-1))
        return inpt.reshape(og_shape)

    def forward(self, x):
        """
        Args:
            x: torch LongTensor (B, S)
                a sequence of indices
        Returns:
            fx: torch tensor (B, H)
        """
        x = x.clone()
        x[x<0] = self.null_idx

        embs = self.embeddings(x)
        embs = self.dropout(embs)
        if self.use_count_words == NUMERAL:
            mask = self.get_mask(x, self.STOP, incl_token=True)
            embs = self.lstm_consol(embs, mask)
        if len(embs.shape)==2:
            outputs = self.consolidator(embs)
        elif embs.shape[1]==1:
            outputs = self.consolidator(embs[:,0])
        return self.proj(outputs)

class NullModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: torch Float Tensor (B, S, C, H, W) or (B,C,H,W)
        Returns:
            actn: torch FloatTensor (B, S, A) or (B, A)
            lang: tuple of torch FloatTensors (B, S, L) or (B, L)
        """
        if len(x.shape) == 4:
            return self.step(x)
        else:
            # Action
            actn = torch.zeros(*x.shape[:2], self.actn_size).float()
            # Language
            lang = torch.zeros(*x.shape[:2], self.lang_size).float()
            if x.is_cuda:
                actn.cuda()
                lang.cuda()
            return self.output_fxn(actn), (lang,)

    def step(self, x):
        """
        Args:
            x: torch Float Tensor (B, C, H, W)
        """
        actn = torch.zeros((x[0], self.actn_size)).float()
        lang = torch.zeros((x[0], self.lang_size)).float()
        if self.is_cuda:
            actn = actn.cuda()
            lang = lang.cuda()
        return self.output_fxn(actn), (lang,)

class TestModel(Model):
    """
    This model collects the data argued to the model so as to ensure
    the inputs are exactly as expected for testing purposes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # each observation is turned into a string and stored inside
        # this variable
        self.data_strings = dict()

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: torch Float Tensor (B, S, C, H, W) or (B,C,H,W)
        Returns:
            actn: torch FloatTensor (B, S, A) or (B, A)
            lang: tuple of torch FloatTensors (B, S, L) or (B, L)
        """
        temp = x.reshape(x.shape[0], x.shape[1], -1)
        for i,xxx in enumerate(temp):
            for j,xx in enumerate(xxx):
                s = str(xx)
                if s in self.data_strings:
                    self.data_strings[s].add(i)
                    o = xx.cpu().detach().data.numpy().reshape(x.shape[2:])
                    plt.imshow(o.transpose((1,2,0)).squeeze())
                    plt.savefig("imgs/row{}_samp{}.png".format(i,j))
                else:
                    self.data_strings[s] = {i}

        if len(x.shape) == 4:
            return self.step(x)
        else:
            # Action
            actn = torch.ones(
                *x.shape[:2],
                self.actn_size,
                requires_grad=True).float()
            # Language
            lang = torch.ones(
                *x.shape[:2],
                self.lang_size,
                requires_grad=True).float()
            if x.is_cuda:
                actn = actn.cuda()
                lang = lang.cuda()
            return self.output_fxn(actn*x.sum()), (lang*x.sum(),)

    def step(self, x):
        """
        Args:
            x: torch Float Tensor (B, C, H, W)
        """
        x = x.reshape(len(x), -1)
        actn = torch.ones(
            (x.shape[0], self.actn_size),
            requires_grad=True).float()
        lang = torch.ones(
            (x.shape[0], self.lang_size),
            requires_grad=True).float()
        if x.is_cuda:
            actn = actn.cuda()
            lang = lang.cuda()
        return self.output_fxn(actn*x.sum()), (lang*x.sum(),)


class RandomModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(inpt_shape=None, **kwargs)

    def forward(self, x, dones=None):
        """
        Args:
            x: torch Float Tensor (B, S, C, H, W)
            dones: torch LongTensor (B, S)
        """
        if len(x.shape) == 4:
            return self.step(x)
        else:
            # Action
            actn = torch.zeros(*x.shape[:2], self.actn_size).float()
            rand = torch.randint(
                low=0,
                high=self.actn_size,
                size=(int(np.prod(x.shape[:2])),)
            )
            actn = actn.reshape(int(np.prod(x.shape[:2])), -1)
            actn[torch.arange(len(actn)).long(), rand] = 1

            # Language
            lang = torch.zeros(*x.shape[:2], self.lang_size).float()
            rand = torch.randint(
                low=0,
                high=self.lang_size,
                size=(int(np.prod(x.shape[:2])),)
            )
            lang = lang.reshape(int(np.prod(x.shape[:2])), -1)
            lang[torch.arange(len(lang)).long(), rand] = 1
            if x.is_cuda:
                actn.cuda()
                lang.cuda()
            return self.output_fxn(actn), (lang,)

    def step(self, x):
        """
        Args:
            x: torch Float Tensor (B, C, H, W)
        """
        rand = torch.randint(
            low=0,
            high=self.actn_size,
            size=(len(x),)
        )
        actn = torch.zeros(len(x), self.actn_size).float()
        actn[torch.arange(len(x)).long(), rand] = 1
        rand = torch.randint(
            low=0,
            high=self.lang_size,
            size=(len(x),)
        )
        lang = torch.zeros(len(x), self.lang_size).float()
        lang[torch.arange(len(x)).long(), rand] = 1
        if x.is_cuda:
            actn.cuda()
            lang.cuda()
        return self.output_fxn(actn), (lang,)

class RawVision(Model):
    """
    Use this module to feed the visual input directly into the LSTM.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = MODEL_TYPES.CNN
        self.shapes = [ *self.inpt_shape[-3:] ]
        self.flat_size = int(np.prod(self.inpt_shape[-3:]))
        self.features = NullOp()

    def step(self, x, *args, **kwargs):
        return x

    def forward(self, x, *args, **kwargs):
        return x

class VaryCNN(Model):
    """
    A simple convolutional network with no recurrence.
        conv2d
        bnorm
        relu
        conv2d
        bnorm
        relu
        linear
        bnorm
        relu
        linear
    """
    def __init__(self, feats_only=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = MODEL_TYPES.CNN
        modules = []
        shape = [*self.inpt_shape[-3:]]
        self.shapes = [shape]
        for i in range(len(self.depths)-1):
            # CONV
            modules.append(
                nn.Conv2d(
                    self.depths[i],
                    self.depths[i+1],
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=self.paddings[i]
                )
            )
            # RELU
            modules.append(GaussianNoise(self.conv_noise))
            modules.append(globals()[self.actv_fxn]())
            # Batch Norm
            if self.bnorm:
                modules.append(nn.BatchNorm2d(self.depths[i+1]))
            # Track Activation Shape Change
            shape = update_shape(
                shape, 
                depth=self.depths[i+1],
                kernel=self.kernels[i],
                stride=self.strides[i],
                padding=self.paddings[i]
            )
            self.shapes.append(shape)
        if self.feat_drop_p > 0:
            modules.append(nn.Dropout(self.feat_drop_p))
        self.features = nn.Sequential(*modules)
        self.flat_size = int(np.prod(shape))

        if not feats_only:
            # Make Action MLP
            self.make_actn_dense(inpt_size=self.flat_size)
            self.make_lang_denses(inpt_size=self.flat_size)

    def step(self, x, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            pred: torch Float Tensor (B, K)
        """
        fx = self.features(x)
        fx = fx.reshape(len(fx), -1)
        actn = self.actn_dense(fx)
        langs = []
        for dense in self.lang_denses:
            lang = dense(fx)
            langs.append(lang)
        return self.output_fxn(actn), langs

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
        """
        b,s = x.shape[:2]
        actn, langs = self.step(x.reshape(-1, *x.shape[2:]))
        langs = torch.stack(langs, dim=0).reshape(len(langs), b, s, -1)
        return actn.reshape(b,s,-1), langs

class ViT(Model):
    """
    A Vision Transformer
        conv embedding projection
        cls cat
        positional encoding
        encoder
        return cls
    """

    def __init__(self, feats_only=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = MODEL_TYPES.CNN
        modules = []
        shape = [*self.inpt_shape[-3:]]
        self.shapes = [shape, shape]
        if (self.depths[1]%self.n_heads) != 0:
            ceil = math.ceil( self.depths[1]/self.n_heads )
            self.depths[1] = ceil*self.n_heads
        self.flat_size = self.depths[1]
        modules = [
            nn.Conv2d(
                self.depths[0],
                self.depths[1],
                kernel_size=self.kernels[0],
                stride=self.strides[0],
                padding=self.paddings[0]
            )
        ]
        modules.append(Reshape((self.depths[1], -1)))
        modules.append(Transpose((0,2,1)))
        self.emb_proj = nn.Sequential(*modules)

        # Transformer
        self.cls = nn.Parameter(
            torch.randn(1,1,self.depths[1])/np.sqrt(self.depths[1])
        )
        self.pos_enc = PositionalEncoding(
            self.depths[1],
            self.feat_drop_p
        )
        enc_layer = nn.TransformerEncoderLayer(
            self.depths[1],
            self.n_heads,
            3*self.depths[1],
            self.feat_drop_p,
            norm_first=True,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            self.n_vit_layers
        )

        if not feats_only:
            self.make_actn_dense()
            self.make_lang_denses()

        # Memory
        if self.lnorm:
            self.layernorm = nn.LayerNorm(self.depths[1])
        self.h = None
        self.c = None
        self.reset(batch_size=1)

    def features(self, x, cls=None, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, C, H, W)
            cls: torch FloatTensor (B,1,D)
        Returns:
            fx: torch FloatTensor (B, D)
        """
        embs = self.emb_proj(x)
        if cls is None: cls = self.cls.repeat((len(x), 1,1))
        embs = torch.cat([cls, embs], axis=1)
        embs = self.pos_enc(embs)
        embs = self.encoder(embs)
        return embs[:,0]

    def step(self, x, cls=None, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
            cls: torch FloatTensor (B,1,D)
        Returns:
            pred: torch FloatTensor (B, D)
        """
        if cls is None:
            cls = self.cls.repeat((len(x),1,1))
        fx = self.features(x, cls)
        actn = self.actn_dense(fx)
        langs = []
        for dense in self.lang_denses:
            lang = dense(fx)
            langs.append(lang)
        return self.output_fxn(actn), langs

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
        """
        b,s = x.shape[:2]
        actn, langs = self.step(x.reshape(-1, *x.shape[2:]))
        langs = torch.stack(langs, dim=0).reshape(len(langs), b, s, -1)
        return actn.reshape(b,s,-1), langs

class SimpleCNN(VaryCNN):
    """
    A simple convolutional network with no recurrence.
        conv2d
        bnorm
        relu
        conv2d
        bnorm
        relu
        linear
        bnorm
        relu
        linear
    """
    def __init__(self, *args, **kwargs):
        kwargs = {
            **kwargs,
            "depths":[32, 48],
            "kernels":[3, 3],
            "strides":[1, 1],
            "paddings":[0, 0],
            "h_mult":2,
        }
        super().__init__( *args, **kwargs )
        self.model_type = MODEL_TYPES.CNN
        modules = []
        shape = [*self.inpt_shape[-3:]]
        self.shapes = [shape]
        for i in range(len(self.depths)-1):
            # CONV
            modules.append(
                nn.Conv2d(
                    self.depths[i],
                    self.depths[i+1],
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=self.paddings[i]
                )
            )
            # RELU
            modules.append(GaussianNoise(self.conv_noise))
            modules.append(nn.ReLU())
            # Batch Norm
            if self.bnorm:
                modules.append(nn.BatchNorm2d(self.depths[i+1]))
            # Track Activation Shape Change
            shape = update_shape(
                shape, 
                depth=self.depths[i+1],
                kernel=self.kernels[i],
                stride=self.strides[i],
                padding=self.paddings[i]
            )
            self.shapes.append(shape)
        if self.feat_drop_p > 0:
            modules.append(nn.Dropout(self.feat_drop_p))
        self.features = nn.Sequential(*modules)
        self.flat_size = int(np.prod(shape))

        # Make Action MLP
        if self.drop_p > 0:
            modules = [
                Flatten(),
                nn.Linear(self.flat_size, self.h_size),
                nn.Dropout(self.drop_p),
                GaussianNoise(self.dense_noise),
                nn.ReLU()
            ]
        else:
            modules = [
                Flatten(),
                nn.Linear(self.flat_size, self.h_size),
                GaussianNoise(self.dense_noise),
                nn.ReLU()
            ]
        if self.bnorm:
            modules.append(nn.BatchNorm1d(self.h_size))
        self.actn_dense = nn.Sequential(
            *modules,
            nn.Linear(self.h_size, self.actn_size)
        )

        # Make Language MLP
        self.lang_denses = nn.ModuleList([])
        for i in range(self.n_lang_denses):
            if self.drop_p > 0:
                modules = [
                    Flatten(),
                    nn.Linear(self.flat_size, self.h_size),
                    nn.Dropout(self.drop_p),
                    nn.ReLU()
                ]
            else:
                modules = [
                    Flatten(),
                    nn.Linear(self.flat_size, self.h_size),
                    nn.ReLU()
                ]
            if self.bnorm:
                modules.append(nn.BatchNorm1d(self.h_size))
            self.lang_denses.append(nn.Sequential(
                *modules,
                nn.Linear(self.h_size, self.lang_size)
            ))

class VaryLSTM(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.max_char_seq is None: self.max_char_seq = 1

        # Convs
        if self.vision_type is None:
            self.cnn = VaryCNN(*args, feats_only=True, **kwargs)
        else:
            self.cnn = globals()[self.vision_type](
                *args, feats_only=True, **kwargs
            )
        self.shapes = self.cnn.shapes
        self.features = self.cnn.features

        # LSTM
        self.flat_size = self.cnn.flat_size
        # add hsize for the cdtnl vectors
        size = self.flat_size + self.h_size

        # Make LSTMs and lang consolidator.
        # The consolidators are either the identity function or they
        # are a class that assists with processing the language
        # for the next timestep
        self.lang_consolidator = identity
        if self.incl_lang_inpt:
            # adding another hsize because we will convert language
            # preds into embeddings
            size = size + self.h_size
            consolidator_kwargs = {
                "lang_size": self.lang_size,
                "h_size": self.h_size,
                "max_char_seq": self.max_char_seq,
                "STOP": self.STOP,
                "drop_p": self.lang_inpt_drop_p,
                "use_count_words": self.use_count_words,
            }
            self.lang_consolidator = InptConsolidationModule(
                **consolidator_kwargs
            )
        self.lstm = nn.LSTMCell(size, self.h_size)

        self.make_actn_dense()
        self.make_lang_denses()
        # Memory
        if self.lnorm:
            self.layernorm_c = nn.LayerNorm(self.h_size)
            self.layernorm_h = nn.LayerNorm(self.h_size)
        self.h_init = None
        self.c_init = None
        if self.learn_h:
            sqr = self.h_size**2
            self.h_init = nn.Parameter(torch.randn(1,self.h_size)/sqr)
            self.c_init = nn.Parameter(torch.randn(1,self.h_size)/sqr)
        self.h_inits = None
        self.c_inits = None
        self.h = None
        self.c = None
        self.lang = None
        self.reset(batch_size=1)

    def process_lang_preds(self, langs):
        """
        Assists in recording the language prediction that will
        potentially be used in the next time step. Averages over the
        preds, reshapes them and then takes the argmax

        Args:
            langs: list of torch FloatTensors [(B,L), (B,L), ...]
        Returns:
            lang: torch LongTensor (B,M)
                the average over the langs list then the argmax over L
                or L//max_char_seq if using NUMERAL variants
        """
        lang = langs[0].detach().data/len(langs)
        for i in range(1,len(langs)):
            lang = lang + langs[i].detach().data/len(langs)
        lang = torch.argmax(
            lang.reshape(len(self.lang),self.max_char_seq,-1),dim=-1
        ).long()
        return lang

    def reset(self, batch_size=1):
        """
        Resets the memory vectors

        Args:
            batch_size: int
                the size of the incoming batches
        Returns:
            None
        """
        if self.learn_h:
            self.h = self.h_init.repeat(batch_size,1)
            self.c = self.c_init.repeat(batch_size,1)
        else:
            self.h = torch.zeros(batch_size, self.h_size).float()
            self.c = torch.zeros(batch_size, self.h_size).float()
        self.lang = torch.zeros(batch_size, self.max_char_seq).long()
        self.lang[:,0] = self.STOP
        # Ensure memory is on appropriate device
        if self.is_cuda:
            self.h.to(self.get_device())
            self.c.to(self.get_device())
            self.lang.to(self.get_device())
        self.prev_hs = [self.h]
        self.prev_cs = [self.c]
        self.prev_langs = [self.lang]

    def partial_reset(self, dones):
        """
        Uses the done signals to reset appropriate parts of the h and
        c vectors.

        Args:
            dones: torch LongTensor (B,)
                h and c are zeroed along any row in which dones[row]==1
        Returns:
            h: torch FloatTensor (B, H)
            c: torch FloatTensor (B, H)
            lang: torch LongTensor (B,L)
        """
        mask = (1-dones).unsqueeze(-1)
        if self.learn_h:
            d = dones.bool()
            m = mask[:,0].bool()
            h = torch.empty_like(self.h)
            c = torch.empty_like(self.c)
            if torch.any(d):
                h[d] = self.h_init
                c[d] = self.c_init
            if torch.any(m):
                h[m] = self.h[m]
                c[m] = self.c[m]
        else:
            h = self.h*mask
            c = self.c*mask
        lang = self.lang*mask
        lang[dones.bool(),0] = self.STOP
        return h,c,lang

    def reset_to_step(self, step=0):
        """
        This function resets all recurrent states in a model to the
        previous recurrent state just after the argued step. So, the
        model takes the 0th step then the 0th h and c vectors are the
        h and c vectors just after the model took this step.

        Args:
            step: int
                the index of the step to revert the recurrence to
        """
        assert step < len(self.prev_hs), "invalid step"
        self.h = self.prev_hs[step].detach().data
        self.c = self.prev_cs[step].detach().data
        self.lang = self.prev_langs[step].detach().data
        if self.is_cuda:
            self.h.to(self.get_device())
            self.c.to(self.get_device())
            self.lang.to(self.get_device())
        self.prev_hs = self.prev_hs[:step+1]
        self.prev_cs = self.prev_cs[:step+1]
        self.prev_langs = self.prev_langs[:step+1]

    def step(self, x, cdtnl, lang_inpt=None, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
                a single step of observations
            cdtnl: torch FloatTensor (B, E)
                the conditional latent vectors
            lang_inpt: None or torch LongTensor (B, M)
                a sequence of language indicies that will be consolidated
                and used as input to the lstm. for non NUMERAL models,
                M will be equal to 1.
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor (B, L)
        """
        self.h = self.h.to(self.get_device())
        self.c = self.c.to(self.get_device())
        fx = self.features(x)
        fx = fx.reshape(len(x), -1) # (B, N)
        inpt = [fx, cdtnl]
        if self.incl_lang_inpt:
            if lang_inpt is None:
                inpt.append(torch.zeros(
                    (len(fx), self.h_size),
                    device=self.get_device())
                )
                cat = torch.cat(inpt, dim=-1)
                h, _ = self.lstm( cat, (self.h, self.c) )
                if self.lnorm:
                    h = self.layernorm_h(h)
                langs = []
                for dense in self.lang_denses:
                    langs.append(dense(h))
                lang = self.process_lang_preds(langs)
                lang_inpt = self.lang_consolidator(lang)
            if self.bottleneck:
                inpt = [
                  torch.zeros_like(fx), torch.zeros_like(cdtnl), lang_inpt
                ]
            else:
                inpt = [fx, cdtnl, lang_inpt]
        cat = torch.cat(inpt, dim=-1)
        self.h, self.c = self.lstm( cat, (self.h, self.c) )
        if self.lnorm:
            self.c = self.layernorm_c(self.c)
            self.h = self.layernorm_h(self.h)
        langs = []
        for dense in self.lang_denses:
            langs.append(dense(self.h))
        self.lang = self.process_lang_preds(langs)
        return self.output_fxn(self.actn_dense(self.h)), langs

    def forward(self, x, dones, tasks, lang_inpts=None, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
            dones: torch Long Tensor (B, S)
                the done signals for the environment. the h and c
                vectors are reset when encountering a done signal
            tasks: torch Long Tensor (B, S)
                the task signal corresponding to each environment.
            lang_inpts: None or LongTensor (B,S,M)
                if not None and self.incl_lang_inpt is true, lang_inpts
                are used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
            langs: torch FloatTensor (N,B,S,L)
        """
        cdtnl = self.cdtnl_lstm(self.cdtnl_idxs)
        seq_len = x.shape[1]
        actns = []
        langs = []
        self.prev_hs = []
        self.prev_cs = []
        self.prev_langs = []
        dones = dones.to(self.get_device())
        for s in range(seq_len):
            if lang_inpts is not None: l = lang_inpts[:,s]
            else: l = None
            actn, lang = self.step(x[:,s],cdtnl[tasks[:,s]],lang_inpts=l)
            actns.append(actn.unsqueeze(1))
            if self.n_lang_denses == 1:
                lang = lang[0].unsqueeze(0).unsqueeze(2) # (1, B, 1, L)
            else:
                lang = torch.stack(lang, dim=0).unsqueeze(2)# (N, B, 1, L)
            langs.append(lang)
            self.h, self.c, self.lang = self.partial_reset(dones[:,s])
            self.prev_hs.append(self.h.detach().data)
            self.prev_cs.append(self.c.detach().data)
            self.prev_langs.append(self.lang.detach().data)
        return ( torch.cat(actns, dim=1), torch.cat(langs, dim=2) )

class LSTMOffshoot(VaryLSTM):
    """
    Class to share functions between DoubleVaryLSTM and SymmetricLSTM
    """
    def get_inits(self):
        """
        Assists in creating the h and c initialization parameter lists.
        """
        sqr = self.h_size**2
        self.h_inits = [
            torch.randn(1,self.h_size)/sqr for _ in range(self.n_lstms)
        ]
        self.h_inits = nn.ParameterList(
            [nn.Parameter(h) for h in self.h_inits]
        )

        self.c_inits = [
            torch.randn(1,self.h_size)/sqr for _ in range(self.n_lstms)
        ]
        self.c_inits = nn.ParameterList(
            [nn.Parameter(c) for c in self.c_inits]
        )

    def get_vector_list(self, n, bsize, vsize, inits=None):
        """
        Returns a list of vectors of dimensions (bsize, vsize) on the
        appropriate device.

        Args:
            n: int
                the number of vectors
            bsize: int
                the batchsize
            vsize: int
                the vector size (dimensionality of each vector)
            inits: list of Parameters or None
                optional list of initialization vectors if learning
                the hidden state
        Returns:
            vecs: list of torch FloatTensor [N, (B,V)]
                a list of length N with vectors of shape (B,V) on
                the appropriate device
        """
        vecs = []
        for i in range(n):
            if inits is not None:
                vecs.append(inits[i].repeat(bsize,1))
            else:
                vecs.append(torch.zeros(bsize, vsize).float())
        if self.is_cuda and inits is None:
            for i in range(n):
                vecs[i] = vecs[i].to(self.get_device())
        return vecs

    def reset(self, batch_size=1):
        """
        Resets the memory vectors

        Args:
            batch_size: int
                the size of the incoming batches
        Returns:
            None
        """
        self.hs = self.get_vector_list(
            self.n_lstms, batch_size, self.h_size, self.h_inits
        )
        self.cs = self.get_vector_list(
            self.n_lstms, batch_size, self.h_size, self.c_inits
        )
        self.lang = torch.zeros(batch_size, self.max_char_seq).long()
        self.lang[:,0] = self.STOP
        if self.is_cuda:
            self.lang = self.lang.to(self.get_device())

        self.prev_hs = [self.hs]
        self.prev_cs = [self.cs]
        self.prev_langs = [self.lang]

    def partial_reset(self, dones):
        """
        Uses the done signals to reset appropriate parts of the h and
        c vectors.

        Args:
            dones: torch LongTensor (B,)
                h and c are zeroed along any row in which dones[row]==1
        Returns:
            h: torch FloatTensor (B, H)
            c: torch FloatTensor (B, H)
        """
        mask = (1-dones).unsqueeze(-1)
        if self.learn_h:
            hs = []
            cs = []
            d = dones.bool()
            m = mask[:,0].bool()
            for i in range(len(self.hs)):
                h = torch.empty_like(self.hs[i])
                c = torch.empty_like(self.cs[i])
                if torch.any(d):
                    h[d] = self.h_inits[i]
                    c[d] = self.c_inits[i]
                if torch.any(m):
                    h[m] = self.hs[i][m]
                    c[m] = self.cs[i][m]
                hs.append(h)
                cs.append(c)
        else:
            hs = [h*mask for h in self.hs]
            cs = [c*mask for c in self.cs]
        lang = self.lang*mask
        lang[dones.bool(),0] = self.STOP
        return hs,cs,lang

    def reset_to_step(self, step=0):
        """
        This function resets all recurrent states in a model to the
        previous recurrent state just after the argued step. So, the
        model takes the 0th step then the 0th h and c vectors are the
        h and c vectors just after the model took this step.

        Args:
            step: int
                the index of the step to revert the recurrence to
        """
        assert step < len(self.prev_hs), "invalid step"
        self.hs = self.prev_hs[step]
        self.cs = self.prev_cs[step]
        d = self.get_device()
        if self.is_cuda:
            self.hs = [h.detach().data.to(d) for h in self.hs]
            self.cs = [c.detach().data.to(d) for c in self.cs]
            self.lang = self.prev_langs[step].detach().data.to(d)
        else:
            self.hs = [h.detach().data for h in self.hs]
            self.cs = [c.detach().data for c in self.cs]
            self.lang = self.prev_langs[step].detach().data
        self.prev_hs = self.prev_hs[:step+1]
        self.prev_cs = self.prev_cs[:step+1]
        self.prev_langs = self.prev_langs[:step+1]

class SeparateLSTM(LSTMOffshoot):
    """
    A model with three LSTMs total. One for language that operates
    independently. Then two sequential LSTMs for the actions that
    receives a cut-gradient language input.
    """
    def __init__(self, *args, **kwargs):
        """
        """
        if "incl_lang_inpt" in kwargs and not kwargs["incl_lang_inpt"]:
            kwargs["incl_lang_inpt"] = True
            print("Setting incl_lang_inpt to true. "+\
                    "SeparateLSTM always includes lang input")
        super().__init__(*args, **kwargs)
        if not self.incl_lang_inpt:
            print("SeparateLSTM always includes lang input")
        self.n_lstms = 3
        self.n_lang_denses = 1

        size = self.flat_size + self.h_size # add hsize for condtional
        self.lang_lstm = nn.LSTMCell(size, self.h_size)

        # Change existing lstm to proper input sizing
        if self.bottleneck:
            self.lstm = nn.LSTMCell(self.h_size, self.h_size)

        size = self.h_size
        if self.skip_lstm:
            assert not self.bottleneck,\
                        "bottleneck and skip_lstm are incompatible"
            # add additional h_size here for conditional input
            size = self.h_size + size + self.flat_size
        self.actn_lstm = nn.LSTMCell(size, self.h_size)

        if self.lang_lnorm:
            self.h_lang_lnorm = nn.LayerNorm(self.h_size)
            if self.c_lnorm:
                self.c_lang_lnorm = nn.LayerNorm(self.h_size)

        if self.learn_h:
            self.get_inits()
        self.reset(1)

        self.make_actn_dense()
        self.make_lang_denses()

    def step(self, x, cdtnl, mask=None,lang_inpt=None,*args,**kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
                a single step of observations
            cdtnl: torch FloatTensor (B, E)
                the conditional latent vectors
            mask: torch LongTensor or BoolTensor (B,)
                Used to avoid continuing calculations in the sequence
                when the sequence is over. Ones in the mask denote
                that the time step should be ignored, zeros should
                be included.
            lang_inpt: None or LongTensor (B,M)
                if not None and self.incl_lang_inpt is true, lang_inpt
                is used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor [ (B, L) ]
        """
        # Not that the mask gets flipped to where 1 means keep the
        # calculation in the data
        if mask is None:
            mask = torch.ones(x.shape[0]).long()
        else:
            mask = (1-mask)
        mask = mask.bool() # bool is important for using the mask as an
        # indexing tool
        device = self.get_device()
        self.lang = self.lang.to(device)
        mask = mask.to(device)
        for i in range(self.n_lstms):
            self.hs[i] = self.hs[i].to(device)
            self.cs[i] = self.cs[i].to(device)

        fx = self.features(x[mask])
        fx = fx.reshape(len(fx), -1) # (B, N)
        inpt = [fx, cdtnl[mask]]
        cat = torch.cat(inpt, dim=-1)

        h,c = (self.hs[2][mask], self.cs[2][mask])
        if self.lang_lnorm:
            h = self.h_lang_lnorm(h)
            if self.c_lnorm:
                c = self.c_lang_lnorm(c)
        lang_h, lang_c = self.lang_lstm(
            cat, (h,c)
        )
        lang = self.lang_denses[0](lang_h)

        if lang_inpt is None:
            lang_inpt = self.process_lang_preds([lang])
        else:
            lang_inpt = lang_inpt[mask]
        lang_inpt = self.lang_consolidator( lang_inpt )

        if self.bottleneck:
            cat = lang_inpt
        else:
            inpt.append(lang_inpt)
            cat = torch.cat(inpt, dim=-1)

        h, c = self.lstm( cat, (self.hs[0][mask], self.cs[0][mask]) )
        if self.lnorm:
            h = self.layernorm_h(h)
            if self.c_lnorm:
                c = self.layernorm_c(c)

        inpt = h
        if self.skip_lstm: 
            inpt = [cat]
            inpt = torch.cat(inpt, dim=-1)

        actn_h, actn_c = self.actn_lstm(
            inpt, (self.hs[1][mask], self.cs[1][mask])
        )
        actn = self.actn_dense(actn_h)

        temp_hs = [h, actn_h, lang_h]
        temp_cs = [c, actn_c, lang_c]
        for i in range(len(self.hs)):
            temp_h = torch.zeros_like(self.hs[i])
            temp_h[mask] = temp_hs[i]
            self.hs[i] = temp_h
            temp_c = torch.zeros_like(self.cs[i])
            temp_c[mask] = temp_cs[i]
            self.cs[i] = temp_c

        actn = self.output_fxn(actn)
        temp_actn = torch.zeros(x.shape[0],actn.shape[-1],device=device)
        temp_lang = torch.zeros(x.shape[0],lang.shape[-1],device=device)
        temp_actn[mask] = actn
        temp_lang[mask] = lang
        return temp_actn, [temp_lang]

    def forward(self,x,dones,tasks,masks,lang_inpts=None,*args,**kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
            dones: torch Long Tensor (B, S)
                the done signals for the environment. the h and c
                vectors are reset when encountering a done signal
            tasks: torch Long Tensor (B, S)
                the task signal corresponding to each environment.
            masks: torch LongTensor or BoolTensor (B,S)
                Used to avoid continuing calculations in the sequence
                when the sequence is over. Ones in the mask denote
                that the time step should be ignored, zeros should
                be included.
            lang_inpts: None or LongTensor (B,S,M)
                if not None and self.incl_lang_inpt is true, lang_inpts
                are used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
            langs: torch FloatTensor (N,B,S,L)
        """
        cdtnl = self.cdtnl_lstm(self.cdtnl_idxs)
        seq_len = x.shape[1]
        actns = []
        langs = []
        self.prev_hs = []
        self.prev_cs = []
        self.prev_langs = []
        if x.is_cuda:
            dones = dones.to(x.get_device())
        for s in range(seq_len):
            if masks[:,s].sum() == len(masks):
                actns.append(torch.zeros_like(actns[-1]))
                langs.append(torch.zeros_like(langs[-1]))
            else:
                lang_inpt=None if lang_inpts is None else lang_inpts[:,s]
                actn, lang = self.step(
                  x[:,s], cdtnl[tasks[:,s]], masks[:,s], lang_inpt
                )
                actns.append(actn.unsqueeze(1))
                lang = lang[0].unsqueeze(0).unsqueeze(2) # (1, B, 1, L)
                langs.append(lang)

            self.hs, self.cs, self.lang = self.partial_reset(dones[:,s])
            self.prev_hs.append([h.detach().data for h in self.hs])
            self.prev_cs.append([c.detach().data for c in self.cs])
            self.prev_langs.append(self.lang)
        return ( torch.cat(actns, dim=1), torch.cat(langs, dim=2) )


class NSepLSTM(SeparateLSTM):
    """
    This is a generalization of the SeparateLSTM. n_lstms determines how
    many LSTMs are used for the policy network. In all cases, only one
    LSTM is used for language generation. The language output is used
    to select an embedding that is fed into the first LSTM in the policy
    network. No gradients are backpropagated into the language lstm from
    the policy network.
    """
    def __init__(self, n_lstms=3, *args, **kwargs):
        """
        Args:
            n_lstms: int
                determines the total number of LSTMs in the model
                including the language lstm
        """
        super().__init__(*args, **kwargs)
        if not self.incl_lang_inpt:
            print("SeparateLSTM variants always include lang input")
        self.n_lstms = n_lstms # The number of LSTMs including the lang lstm 
        self.n_lang_denses = 1
        del self.actn_lstm

        size = self.flat_size + self.h_size # add hsize for condtional
        self.lang_lstm = nn.LSTMCell(size, self.h_size)

        self.lstms = nn.ModuleList([
            self.lstm
        ])
        del self.lstm

        if self.bottleneck:
            self.lstms[0] = nn.LSTMCell(self.h_size, self.h_size)

        for i in range(2, self.n_lstms):
            size = self.h_size
            if self.skip_lstm:
                assert not self.bottleneck,\
                            "bottleneck and skip_lstm are incompatible"
                # add additional h_size here for conditional input
                size = self.h_size + size + self.flat_size
            self.lstms.append( nn.LSTMCell(size, self.h_size) )

        if self.learn_h:
            self.get_inits()
        self.reset(1)

        self.make_actn_dense()
        self.make_lang_denses()

    def step(self, x, cdtnl, mask=None,lang_inpt=None,*args,**kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
                a single step of observations
            cdtnl: torch FloatTensor (B, E)
                the conditional latent vectors
            mask: torch LongTensor or BoolTensor (B,)
                Used to avoid continuing calculations in the sequence
                when the sequence is over. Ones in the mask denote
                that the time step should be ignored, zeros should
                be included.
            lang_inpt: None or LongTensor (B,M)
                if not None and self.incl_lang_inpt is true, lang_inpt
                is used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor [ (B, L) ]
        """
        # Not that the mask gets flipped to where 1 means keep the
        # calculation in the data
        if mask is None:
            mask = torch.ones(x.shape[0]).long()
        else:
            mask = (1-mask)
        mask = mask.bool() # bool is important for using the mask as an
        # indexing tool
        device = self.get_device()
        self.lang = self.lang.to(device)
        mask = mask.to(device)
        for i in range(self.n_lstms):
            self.hs[i] = self.hs[i].to(device)
            self.cs[i] = self.cs[i].to(device)

        fx = self.features(x[mask])
        fx = fx.reshape(len(fx), -1) # (B, N)
        inpt = [fx, cdtnl[mask]]
        cat = torch.cat(inpt, dim=-1)

        h,c = (self.hs[-1][mask], self.cs[-1][mask])
        if self.lang_lnorm:
            h = self.h_lang_lnorm(h)
            if self.c_lnorm:
                c = self.c_lang_lnorm(c)
        lang_h, lang_c = self.lang_lstm(
            cat, (h,c)
        )
        lang = self.lang_denses[0](lang_h)

        if lang_inpt is None:
            lang_inpt = self.process_lang_preds([lang])
        else:
            lang_inpt = lang_inpt[mask]
        lang_inpt = self.lang_consolidator( lang_inpt )

        if self.bottleneck:
            cat = lang_inpt
        else:
            inpt.append(lang_inpt)
            cat = torch.cat(inpt, dim=-1)

        h, c = self.lstms[0]( cat, (self.hs[0][mask], self.cs[0][mask]) )
        if self.lnorm:
            h = self.layernorm_h(h)
            if self.c_lnorm:
                c = self.layernorm_c(c)
        hs = [h]
        cs = [c]

        for i in range(1, len(self.lstms)):
            inpt = h
            if self.skip_lstm: 
                inpt = [inpt, cat]
                inpt = torch.cat(inpt, dim=-1)

            h, c = self.lstms[i](
                inpt, (self.hs[i][mask], self.cs[i][mask])
            )
            hs.append(h)
            cs.append(c)

        actn = self.actn_dense(hs[-1])

        hs.append(lang_h)
        cs.append(lang_c)

        for i in range(len(self.hs)):
            temp_h = torch.zeros_like(self.hs[i])
            temp_h[mask] = hs[i]
            self.hs[i] = temp_h
            temp_c = torch.zeros_like(self.cs[i])
            temp_c[mask] = cs[i]
            self.cs[i] = temp_c

        actn = self.output_fxn(actn)
        temp_actn = torch.zeros(x.shape[0],actn.shape[-1],device=device)
        temp_lang = torch.zeros(x.shape[0],lang.shape[-1],device=device)
        temp_actn[mask] = actn
        temp_lang[mask] = lang
        return temp_actn, [temp_lang]


class SymmetricLSTM(LSTMOffshoot):
    """
    A model with three LSTMs total. One for a "core cognition" system,
    and one for each the language and action outputs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_lstms = 3
        self.n_lang_denses = 1

        size = self.h_size
        if self.skip_lstm:
            # add additional h_size here for conditional input
            size = self.h_size + size
        self.actn_lstm = nn.LSTMCell(size, self.h_size)
        self.lang_lstm = nn.LSTMCell(size, self.h_size)
        if self.learn_h:
            self.get_inits()
        self.reset(1)

        self.make_actn_dense()
        self.make_lang_denses()

    def step(self, x, cdtnl, mask=None,lang_inpt=None,*args,**kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
                a single step of observations
            cdtnl: torch FloatTensor (B, E)
                the conditional latent vectors
            mask: torch LongTensor or BoolTensor (B,)
                Used to avoid continuing calculations in the sequence
                when the sequence is over. Ones in the mask denote
                that the time step should be ignored, zeros should
                be included.
            lang_inpt: None or LongTensor (B,M)
                if not None and self.incl_lang_inpt is true, lang_inpts
                are used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor [ (B, L) ]
        """
        # Not that the mask gets flipped to where 1 means keep the
        # calculation in the data
        if mask is None:
            mask = torch.ones(x.shape[0]).long()
        else:
            mask = (1-mask)
        mask = mask.bool() # bool is important for using the mask as an
        # indexing tool
        device = self.get_device()
        self.lang = self.lang.to(device)
        mask = mask.to(device)
        for i in range(self.n_lstms):
            self.hs[i] = self.hs[i].to(device)
            self.cs[i] = self.cs[i].to(device)

        fx = self.features(x[mask])
        fx = fx.reshape(len(fx), -1) # (B, N)
        inpt = [fx, cdtnl[mask]]
        if self.incl_lang_inpt:
            prev_lang = self.lang if lang_inpt is None else lang_inpt
            lang_inpt = self.lang_consolidator(prev_lang[mask])
            inpt.append(lang_inpt)
        cat = torch.cat(inpt, dim=-1)

        h, c = self.lstm( cat, (self.hs[0][mask], self.cs[0][mask]) )
        if self.lnorm:
            if self.c_lnorm:
                c = self.layernorm_c(c)
            h = self.layernorm_h(h)

        inpt = h
        if self.skip_lstm: 
            inpt = [cat]
            inpt = torch.cat(inpt, dim=-1)

        actn_h, actn_c = self.actn_lstm(
            inpt, (self.hs[1][mask], self.cs[1][mask])
        )
        actn = self.actn_dense(actn_h)

        lang_h, lang_c = self.lang_lstm(
            inpt, (self.hs[2][mask], self.cs[2][mask])
        )
        lang = self.lang_denses[0](lang_h)

        temp_hs = [h, actn_h, lang_h]
        temp_cs = [c, actn_c, lang_c]
        for i in range(len(self.hs)):
            temp_h = torch.zeros_like(self.hs[i])
            temp_h[mask] = temp_hs[i]
            self.hs[i] = temp_h
            temp_c = torch.zeros_like(self.cs[i])
            temp_c[mask] = temp_cs[i]
            self.cs[i] = temp_c

        actn = self.output_fxn(actn)
        temp_actn = torch.zeros(x.shape[0],actn.shape[-1],device=device)
        temp_lang = torch.zeros(x.shape[0],lang.shape[-1],device=device)
        temp_actn[mask] = actn
        temp_lang[mask] = lang
        self.lang = self.process_lang_preds([temp_lang])
        return temp_actn, [temp_lang]

    def forward(self,x,dones,tasks,masks,lang_inpts=None,*args,**kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
            dones: torch Long Tensor (B, S)
                the done signals for the environment. the h and c
                vectors are reset when encountering a done signal
            tasks: torch Long Tensor (B, S)
                the task signal corresponding to each environment.
            masks: torch LongTensor or BoolTensor (B,S)
                Used to avoid continuing calculations in the sequence
                when the sequence is over. Ones in the mask denote
                that the time step should be ignored, zeros should
                be included.
            lang_inpts: None or LongTensor (B,S,M)
                if not None and self.incl_lang_inpt is true, lang_inpts
                are used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
            langs: torch FloatTensor (N,B,S,L)
        """
        cdtnl = self.cdtnl_lstm(self.cdtnl_idxs)
        seq_len = x.shape[1]
        actns = []
        langs = []
        self.prev_hs = []
        self.prev_cs = []
        self.prev_langs = []
        if x.is_cuda:
            dones = dones.to(x.get_device())
        for s in range(seq_len):
            if masks[:,s].sum() == len(masks):
                actns.append(torch.zeros_like(actns[-1]))
                langs.append(torch.zeros_like(langs[-1]))
            else:
                lang_inpt=None if lang_inpts is None else lang_inpts[:,s]
                actn, lang = self.step(
                  x[:,s], cdtnl[tasks[:,s]], masks[:,s], lang_inpt
                )
                actns.append(actn.unsqueeze(1))
                lang = lang[0].unsqueeze(0).unsqueeze(2) # (1, B, 1, L)
                langs.append(lang)

            self.hs, self.cs, self.lang = self.partial_reset(dones[:,s])
            self.prev_hs.append([h.detach().data for h in self.hs])
            self.prev_cs.append([c.detach().data for c in self.cs])
            self.prev_langs.append(self.lang)
        return ( torch.cat(actns, dim=1), torch.cat(langs, dim=2) )


class DoubleVaryLSTM(LSTMOffshoot):
    """
    A model with two LSTMs. One for each the language and action outputs.
    The structure is such that one LSTM feeds into the other.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_lstms = 2

        self.lstm0 = self.lstm
        size = self.h_size
        if self.skip_lstm: 
            # Multiply h_size by two for the conditional input
            size = self.flat_size+2*self.h_size
        self.lstm1 = nn.LSTMCell(size, self.h_size)

        if self.extra_lang_pred and self.incl_lang_inpt:
            self.n_lstms = 3
            size = self.flat_size+self.h_size
            self.lstm2 = nn.LSTMCell(size, self.h_size)

        if self.learn_h:
            self.get_inits()
        self.reset(1)
        self.make_actn_dense()
        self.make_lang_denses()

    def step(self, x, cdtnl, lang_inpt=None, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
                a single step of observations
            cdtnl: torch FloatTensor (B, E)
                the conditional latent vectors
            lang_inpt: None or LongTensor (B,M)
                if not None and self.incl_lang_inpt is true, lang_inpts
                are used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor (B, L)
        """
        if x.is_cuda and not self.hs[0].is_cuda:
            for i in range(self.n_lstms):
                self.hs[i] = self.hs[i].to(x.get_device())
                self.cs[i] = self.cs[i].to(x.get_device())
        fx = self.features(x)
        fx = fx.reshape(len(x), -1) # (B, N)
        inpt = [fx, cdtnl]

        langs = []
        if self.incl_lang_inpt:
            cat = None
            if lang_inpt is None:
                inpt.append(torch.zeros(
                    (len(fx), self.h_size),
                    device=self.get_device())
                )
                cat = torch.cat(inpt, dim=-1)
                h, _ = self.lstm0( cat, (self.hs[0], self.cs[0]) )
                if self.lnorm:
                    h = self.layernorm_h(h)

                if not self.stagger_preds:
                    inpt = h
                    if self.skip_lstm: inpt=torch.cat([cat,inpt],dim=-1)
                    h, _ = self.lstm1( inpt, (self.hs[1], self.cs[1]) )
                for dense in self.lang_denses:
                    langs.append(dense(h))
                lang = self.process_lang_preds(langs)

            # Need to do this regardless of argued lang_inpts
            # to ensure hidden state vectors get updated
            if self.extra_lang_pred:
                if cat is None: cat = torch.cat(inpt, dim=-1)
                else: cat = cat[:,:-self.h_size].clone()
                h2,c2 = self.lstm2( cat, (self.hs[2],self.cs[2]) )
                temp = []
                for dense in self.lang_denses:
                    temp.append(dense(h2))
                lang = self.process_lang_preds(temp)
                langs = [*langs, *temp]

            lang_inpt = lang if lang_inpt is None else lang_inpt
            lang_inpt = self.lang_consolidator(lang_inpt)
            if self.bottleneck:
                inpt = [
                  torch.zeros_like(fx),torch.zeros_like(cdtnl),lang_inpt
                ]
            else:
                inpt = [fx, cdtnl, lang_inpt]
        cat = torch.cat(inpt, dim=-1)

        h0, c0 = self.lstm0( cat, (self.hs[0], self.cs[0]) )
        if self.lnorm:
            h0 = self.layernorm_h(h0)
            if self.c_lnorm:
                c0 = self.layernorm_c(c0)

        inpt = h0
        if self.skip_lstm: inpt = torch.cat([cat,inpt],dim=-1)
        h1, c1 = self.lstm1( inpt, (self.hs[1], self.cs[1]) )

        if self.stagger_preds:
            lang_in, actn_in = (inpt,h1) if self.lstm_lang_first else (h1,inpt)
        else: lang_in,actn_in = h1,h1
        if len(langs)==0:
            for dense in self.lang_denses:
                langs.append(dense(lang_in))
        actn = self.actn_dense(actn_in)

        self.hs = [h0, h1]
        self.cs = [c0, c1]
        if self.incl_lang_inpt and self.extra_lang_pred:
            self.hs.append(h2)
            self.cs.append(c2)
        self.lang = self.process_lang_preds(langs)
        return self.output_fxn(actn), langs

    def forward(self, x, dones, tasks, lang_inpts=None, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
            dones: torch Long Tensor (B, S)
                the done signals for the environment. the h and c
                vectors are reset when encountering a done signal
            tasks: torch Long Tensor (B, S)
                the task signal corresponding to each environment.
            lang_inpts: None or LongTensor (B,S,M)
                if not None and self.incl_lang_inpt is true, lang_inpts
                are used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
            langs: torch FloatTensor (N,B,S,L)
        """
        cdtnl = self.cdtnl_lstm(self.cdtnl_idxs)
        seq_len = x.shape[1]
        actns = []
        langs = []
        self.prev_hs = []
        self.prev_cs = []
        self.prev_langs = []
        if x.is_cuda:
            dones = dones.to(x.get_device())
        for s in range(seq_len):
            l = None if lang_inpts is None else lang_inpts[:,s]
            actn, lang = self.step(x[:,s], cdtnl[tasks[:,s]], l)
            actns.append(actn.unsqueeze(1))
            if self.n_lang_denses == 1 and not self.extra_lang_pred:
                lang = lang[0].unsqueeze(0).unsqueeze(2) # (1, B, 1, L)
            else:
                lang = torch.stack(lang, dim=0).unsqueeze(2)# (N, B, 1, L)
            langs.append(lang)
            self.hs, self.cs, self.lang = self.partial_reset(dones[:,s])
            self.prev_hs.append([h.detach().data for h in self.hs])
            self.prev_cs.append([c.detach().data for c in self.cs])
            self.prev_langs.append(self.lang.detach().data)
        return ( torch.cat(actns, dim=1), torch.cat(langs, dim=2) )


class DblBtlComboLSTM(LSTMOffshoot):
    """
    A model with three LSTMs. The first, the lang lstm, maps from visual
    input to language prediction. Another, the bottleneck LSTM receives
    this language input only, with stopped gradients. The last, the actn
    lstm, recieves visual input only. Then the hidden state from
    the actn lstm is concatenated with the bottleneck lstm and used
    for action prediction.
    """
    def __init__(self, *args, **kwargs):
        """
        """
        if "incl_lang_inpt" in kwargs and not kwargs["incl_lang_inpt"]:
            kwargs["incl_lang_inpt"] = True
            print("Setting incl_lang_inpt to true. "+\
                    "DblBtlComboLSTM always includes lang input")
        super().__init__(*args, **kwargs)
        if not self.incl_lang_inpt:
            assert False, "DblBtlComboLSTM needs incl_lang_inpt to be true"
        self.n_lstms = 3
        self.n_lang_denses = 1

        del self.lstm
        self.btl_lstm = nn.LSTMCell(self.h_size, self.h_size)

        size = self.flat_size + self.h_size # add hsize for condtional
        # Actn LSTM
        self.actn_lstm = nn.LSTMCell(size, self.h_size)
        # Lang LSTM
        self.lang_lstm = nn.LSTMCell(size, self.h_size)

        if self.lang_lnorm:
            self.h_lang_lnorm = nn.LayerNorm(self.h_size)
            if self.c_lnorm:
                self.c_lang_lnorm = nn.LayerNorm(self.h_size)

        if self.learn_h:
            self.get_inits()
        self.reset(1)

        self.make_actn_dense(inpt_size=2*self.h_size)
        self.make_lang_denses()

    def step(self, x, cdtnl, mask=None,lang_inpt=None,*args,**kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
                a single step of observations
            cdtnl: torch FloatTensor (B, E)
                the conditional latent vectors
            mask: torch LongTensor or BoolTensor (B,)
                Used to avoid continuing calculations in the sequence
                when the sequence is over. Ones in the mask denote
                that the time step should be ignored, zeros should
                be included.
            lang_inpt: None or LongTensor (B,M)
                if not None and self.incl_lang_inpt is true, lang_inpt
                is used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor [ (B, L) ]
        """
        # Not that the mask gets flipped to where 1 means keep the
        # calculation in the data
        if mask is None:
            mask = torch.ones(x.shape[0]).long()
        else:
            mask = (1-mask)
        mask = mask.bool() # bool is important for using the mask as an
        # indexing tool
        device = self.get_device()
        self.lang = self.lang.to(device)
        mask = mask.to(device)
        for i in range(self.n_lstms):
            self.hs[i] = self.hs[i].to(device)
            self.cs[i] = self.cs[i].to(device)

        fx = self.features(x[mask])
        fx = fx.reshape(len(fx), -1) # (B, N)
        inpt = [fx, cdtnl[mask]]
        cat = torch.cat(inpt, dim=-1)

        actn_h, actn_c = self.actn_lstm(
            cat, (self.hs[0][mask], self.cs[0][mask])
        )
        if self.lnorm:
            actn_h = self.layernorm_h(actn_h)
            if self.c_lnorm:
                actn_c = self.layernorm_c(actn_c)


        lang_h, lang_c = self.lang_lstm(
            cat, (self.hs[2][mask], self.cs[2][mask])
        )
        if self.lang_lnorm:
            lang_h = self.h_lang_lnorm(lang_h)
            if self.c_lnorm:
                lang_c = self.c_lang_lnorm(lang_c)
        lang = self.lang_denses[0](lang_h)
        langs = [lang]
        if self.extra_lang_pred:
            langs.append(self.lang_denses[0](actn_h))

        if lang_inpt is None:
            lang_inpt = self.process_lang_preds([lang])
        else:
            lang_inpt = lang_inpt[mask]
        lang_inpt = self.lang_consolidator( lang_inpt )

        btl_h, btl_c = self.btl_lstm( 
            lang_inpt, (self.hs[1][mask], self.cs[1][mask])
        )

        cat = torch.cat([actn_h, btl_h],dim=-1)
        actn = self.actn_dense(cat)

        temp_hs = [actn_h, btl_h, lang_h]
        temp_cs = [actn_c, btl_c, lang_c]
        for i in range(len(self.hs)):
            temp_h = torch.zeros_like(self.hs[i])
            temp_h[mask] = temp_hs[i]
            self.hs[i] = temp_h
            temp_c = torch.zeros_like(self.cs[i])
            temp_c[mask] = temp_cs[i]
            self.cs[i] = temp_c

        # Need to redistribute processed outputs to unmasked state
        actn = self.output_fxn(actn)
        temp_actn = torch.zeros(x.shape[0],actn.shape[-1],device=device)
        temp_actn[mask] = actn
        temp_langs = []
        for lang in langs:
            temp_lang = torch.zeros(x.shape[0],lang.shape[-1],device=device)
            temp_lang[mask] = lang
            temp_langs.append(temp_lang)
        return temp_actn, temp_langs

    def forward(self,x,dones,tasks,masks,lang_inpts=None,*args,**kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
            dones: torch Long Tensor (B, S)
                the done signals for the environment. the h and c
                vectors are reset when encountering a done signal
            tasks: torch Long Tensor (B, S)
                the task signal corresponding to each environment.
            masks: torch LongTensor or BoolTensor (B,S)
                Used to avoid continuing calculations in the sequence
                when the sequence is over. Ones in the mask denote
                that the time step should be ignored, zeros should
                be included.
            lang_inpts: None or LongTensor (B,S,M)
                if not None and self.incl_lang_inpt is true, lang_inpts
                are used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
            langs: torch FloatTensor (N,B,S,L)
        """
        cdtnl = self.cdtnl_lstm(self.cdtnl_idxs)
        seq_len = x.shape[1]
        actns = []
        langs = []
        self.prev_hs = []
        self.prev_cs = []
        self.prev_langs = []
        if x.is_cuda:
            dones = dones.to(x.get_device())
        for s in range(seq_len):
            if masks[:,s].sum() == len(masks):
                actns.append(torch.zeros_like(actns[-1]))
                langs.append(torch.zeros_like(langs[-1]))
            else:
                lang_inpt=None if lang_inpts is None else lang_inpts[:,s]
                actn, lang = self.step(
                  x[:,s], cdtnl[tasks[:,s]], masks[:,s], lang_inpt
                )
                actns.append(actn.unsqueeze(1))
                lang = torch.stack(lang, dim=0).unsqueeze(2)# (N, B, 1, L)
                langs.append(lang)

            self.hs, self.cs, self.lang = self.partial_reset(dones[:,s])
            self.prev_hs.append([h.detach().data for h in self.hs])
            self.prev_cs.append([c.detach().data for c in self.cs])
            self.prev_langs.append(self.lang)
        return ( torch.cat(actns, dim=1), torch.cat(langs, dim=2) )


class NVaryLSTM(DoubleVaryLSTM):
    """
    A model with N LSTMs. 
    The structure is such that each LSTM feeds into the next.
    """
    def __init__(self, n_lstms=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_lstms = n_lstms
        try:
            del self.lstm0
            del self.lstm1
        except: pass

        self.lstms = nn.ModuleList([])
        self.lstms.append(self.lstm)
        self.h_lnorms = nn.ModuleList([])
        self.h_lnorms.append(self.layernorm_h)
        self.c_lnorms = nn.ModuleList([])
        self.c_lnorms.append(self.layernorm_c)

        size = self.h_size
        if self.skip_lstm: 
            # Multiply h_size by two for the conditional input
            size = self.flat_size+2*self.h_size
        for _ in range(self.n_lstms-1):
            self.lstms.append(nn.LSTMCell(size, self.h_size))
            #self.h_lnorms.append(nn.LayerNorm(self.h_size))
            #self.c_lnorms.append(nn.LayerNorm(self.c_size))
        if self.learn_h: self.get_inits()
        self.reset(1)
        self.make_actn_dense()
        self.make_lang_denses()

    def step(self, x, cdtnl, lang_inpt=None, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
                a single step of observations
            cdtnl: torch FloatTensor (B, E)
                the conditional latent vectors
            lang_inpt: None or LongTensor (B,M)
                if not None and self.incl_lang_inpt is true, lang_inpts
                are used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor (B, L)
        """
        if x.is_cuda:
            for i in range(self.n_lstms):
                self.hs[i] = self.hs[i].to(x.get_device())
                self.cs[i] = self.cs[i].to(x.get_device())
        fx = self.features(x)
        fx = fx.reshape(len(x), -1) # (B, N)
        inpt = [fx, cdtnl]

        if self.incl_lang_inpt:
            raise NotImplemented
        cat = torch.cat(inpt, dim=-1)

        hs = []
        cs = []
        h, c = self.lstms[0]( cat, (self.hs[0], self.cs[0]) )
        if self.lnorm:
            h = self.h_lnorms[0](h)
            if self.c_lnorm:
                c = self.c_lnorms[0](c)
        hs.append(h)
        cs.append(c)

        for i in range(1,self.n_lstms):
            inpt = h
            if self.skip_lstm: inpt = torch.cat([cat,inpt],dim=-1)
            h, c = self.lstms[i]( inpt, (self.hs[1], self.cs[1]) )
            hs.append(h)
            cs.append(c)

        lang_in, actn_in = (hs[-1],hs[-1])
        if self.stagger_preds and self.n_lstms>1:
            lang_in, actn_in = (hs[-2],hs[-1]) if self.lstm_lang_first\
                                               else (hs[-1],hs[-2])
        langs = []
        for dense in self.lang_denses:
            langs.append(dense(lang_in))
        actn = self.actn_dense(actn_in)

        self.hs = [*hs]
        self.cs = [*cs]
        self.lang = self.process_lang_preds(langs)
        return self.output_fxn(actn), langs


class SimpleLSTM(VaryLSTM):
    """
    A recurrent LSTM model.
    """
    def __init__(self, *args, **kwargs):
        kwargs = {
            **kwargs,
            "depths":[32, 48],
            "kernels":[3, 3],
            "strides":[1, 1],
            "paddings":[0, 0],
            "h_mult":2,
            "skip_lstm": False,
        }
        super().__init__( *args, **kwargs )
        # Action Dense
        if self.drop_p > 0:
            self.actn_dense = nn.Sequential(
                nn.Dropout(self.drop_p),
                GaussianNoise(self.dense_noise),
                nn.ReLU(),
                nn.Linear(self.h_size, self.actn_size),
            )
        else:
            self.actn_dense = nn.Sequential(
                GaussianNoise(self.dense_noise),
                nn.ReLU(),
                nn.Linear(self.h_size, self.actn_size),
            )

        # Lang Dense
        self.lang_denses = nn.ModuleList([])
        for i in range(self.n_lang_denses):
            if self.drop_p > 0:
                self.lang_denses.append(nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.h_size, 2*self.h_size),
                    nn.Dropout(self.drop_p),
                    nn.ReLU(),
                    nn.Linear(2*self.h_size, self.lang_size),
                ))
            else:
                self.lang_denses.append(nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.h_size, 2*self.h_size),
                    nn.ReLU(),
                    nn.Linear(2*self.h_size, self.lang_size),
                ))

class NoConvLSTM(VaryLSTM):
    """
    An LSTM that only uses two dense layers as the preprocessing of the
    image before input to the recurrence. Instead of a convolutional
    vision module, we use a single layer MLP
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flat_size = int(np.prod(self.inpt_shape[-3:]))
        modules = [Flatten()]
        if self.lnorm: modules.append(nn.LayerNorm(self.flat_size))
        modules.append(nn.Linear(self.flat_size, self.flat_size))
        modules.append(nn.ReLU())
        if self.lnorm: modules.append(nn.LayerNorm(self.flat_size))
        modules.append(nn.Linear(self.flat_size, self.flat_size))
        if self.feat_drop_p > 0:
            modules.append(nn.Dropout(self.feat_drop_p))
        modules.append(nn.ReLU())
        self.features = nn.Sequential(*modules)

        self.lstm = nn.LSTMCell(self.flat_size, self.h_size)

class DoubleLSTM(DoubleVaryLSTM):
    """
    A recurrent LSTM model.
    """
    def __init__(self, *args, **kwargs):
        kwargs = {
            **kwargs,
            "depths":[32, 48],
            "kernels":[3, 3],
            "strides":[1, 1],
            "paddings":[0, 0],
            "h_mult":2,
            "skip_lstm": False,
        }
        super().__init__( *args, **kwargs )
        if self.drop_p > 0:
            self.actn_dense = nn.Sequential(
                nn.Dropout(self.drop_p),
                GaussianNoise(self.dense_noise),
                nn.ReLU(),
                nn.Linear(self.h_size, self.actn_size),
            )
        else:
            self.actn_dense = nn.Sequential(
                GaussianNoise(self.dense_noise),
                nn.ReLU(),
                nn.Linear(self.h_size, self.actn_size),
            )

        # Lang Dense
        self.lang_denses = nn.ModuleList([])
        for i in range(self.n_lang_denses):
            if self.drop_p > 0:
                self.lang_denses.append(nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.h_size, 2*self.h_size),
                    nn.Dropout(self.drop_p),
                    nn.ReLU(),
                    nn.Linear(2*self.h_size, self.lang_size),
                ))
            else:
                self.lang_denses.append(nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.h_size, 2*self.h_size),
                    nn.ReLU(),
                    nn.Linear(2*self.h_size, self.lang_size),
                ))

class Transformer(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = MODEL_TYPES.TRANSFORMER

        # Convs
        if self.vision_type is None:
            self.cnn = VaryCNN(*args, feats_only=True, **kwargs)
        else:
            self.cnn = globals()[self.vision_type](
                *args, feats_only=True, **kwargs
            )
        self.shapes = self.cnn.shapes
        self.features = self.cnn.features

        # Linear Projection
        self.flat_size = self.cnn.flat_size
        self.proj = nn.Sequential(
            Flatten(),
            nn.Linear(self.flat_size, self.h_size)
        )

        # Transformer
        self.pos_enc = PositionalEncoding(
            self.h_size,
            self.feat_drop_p
        )
        enc_layer = nn.TransformerEncoderLayer(
            self.h_size,
            self.n_heads,
            3*self.h_size,
            self.feat_drop_p,
            norm_first=True,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            self.n_layers
        )

        self.make_actn_dense()
        self.make_lang_denses()

        # Memory
        if self.lnorm:
            self.layernorm = nn.LayerNorm(self.h_size)
        self.h = None
        self.c = None
        self.hs = []
        self.reset(batch_size=1)
        self.register_buffer(
            "fwd_mask",
            get_transformer_fwd_mask(s=self.max_ctx_len+1)
        )

    def reset(self, batch_size=1):
        """
        Resets the memory vectors

        Args:
            batch_size: int
                the size of the incoming batches
        Returns:
            None
        """
        self.prev_hs = collections.deque(maxlen=self.seq_len)
        self.h = torch.zeros(
            batch_size, self.h_size, device=self.get_device()
        )

    def partial_reset(self, dones):
        """
        Uses the done signals to reset appropriate parts of the h and
        c vectors.

        Args:
            dones: torch LongTensor (B,)
                h and c are zeroed along any row in which dones[row]==1
        Returns:
            h: torch FloatTensor (B, H)
            c: torch FloatTensor (B, H)
        """
        pass

    def reset_to_step(self, step=0):
        """
        This function resets all recurrent states in a model to the
        previous recurrent state just after the argued step. So, the
        model takes the 0th step then the 0th h and c vectors are the
        h and c vectors just after the model took this step.

        Args:
            step: int
                the index of the step to revert the recurrence to
        """
        if len(self.prev_hs) > step:
            self.prev_hs = collections.deque(
                list(self.prev_hs)[:step],
                maxlen=self.seq_len
            )

    def step(self, x, cdtnl, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
            cdtnl: torch FloatTensor (B, E)
                the conditional latent vectors
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor (B, L)
        """
        if len(self.prev_hs)==0: self.prev_hs.append(cdtnl)
        fx = self.features(x)
        fx = self.proj(fx) # (B, H)
        self.prev_hs.append(fx)
        encs = torch.stack(list(self.prev_hs), dim=1)
        encs = self.pos_enc(encs)
        slen = encs.shape[1]
        encs = self.encoder( encs )
        encs = encs[:,-1] # grab last prediction only
        if self.lnorm:
            encs = self.layernorm(encs)
        self.h = encs
        langs = []
        for dense in self.lang_denses:
            langs.append(dense(encs))
        return self.output_fxn(self.actn_dense(encs)), langs

    def forward(self, x, tasks, masks=None, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
            tasks: torch LongTensor (B,S)
            masks: torch LongTensor (B,S)
                Used to remove padding from the attention calculations.
                Ones denote locations that should be ignored. 0s denote
                locations that should be included.
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
            langs: torch FloatTensor (N,B,S,L)
        """
        cdtnl = self.cdtnl_lstm(self.cdtnl_idxs)
        seq_len = x.shape[1]
        self.prev_hs = collections.deque(maxlen=self.seq_len)
        b,s,c,h,w = x.shape
        fx = self.features(x.reshape(-1,c,h,w)).reshape(b*s,-1)
        fx = self.proj(fx).reshape(b,s,-1)
        # Add conditional vector
        fx = torch.cat([cdtnl[tasks[:,0]].unsqueeze(1), fx], dim=1)
        encs = self.pos_enc(fx)
        if masks is not None:
            masks = torch.cat([
                torch.zeros(b,1).to(self.get_device()),
                masks
            ], dim=1).bool() # cat for the conditional
        encs = self.encoder(
            encs,
            mask=self.fwd_mask[:s+1,:s+1],
            src_key_padding_mask=masks
        )
        encs = encs[:,1:] # Remove conditional vector
        if self.lnorm:
            encs = self.layernorm(encs)
        encs = encs.reshape(b*s,-1)
        actns = self.actn_dense(encs).reshape(b,s,-1)
        langs = []
        for dense in self.lang_denses:
            langs.append(dense(encs).reshape(b,s,-1))
        return self.output_fxn(actns), torch.stack(langs,dim=0)

class SepTransformer(Transformer):
    """
    Same as transformer except that there are seperate encoding branches
    for the language and action outputs off the main encoding layers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        enc_layer = nn.TransformerEncoderLayer(
            self.h_size,
            self.n_heads,
            3*self.h_size,
            self.feat_drop_p,
            norm_first=True,
            batch_first=True
        )
        self.lang_encoder = nn.TransformerEncoder( enc_layer, 1 )
        self.actn_encoder = nn.TransformerEncoder( enc_layer, 1 )

        # Memory
        if self.lnorm:
            self.actnnorm = nn.LayerNorm(self.h_size)
            self.langnorm = nn.LayerNorm(self.h_size)
        self.hs = []
        self.reset(batch_size=1)

    def reset(self, batch_size=1):
        """
        Resets the memory vectors

        Args:
            batch_size: int
                the size of the incoming batches
        Returns:
            None
        """
        self.prev_hs = collections.deque(maxlen=self.seq_len)
        self.hs = [
          torch.zeros(batch_size,self.h_size,device=self.get_device()),
          torch.zeros(batch_size,self.h_size,device=self.get_device()),
          torch.zeros(batch_size,self.h_size,device=self.get_device()),
        ]


    def step(self, x, cdtnl, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
            cdtnl: torch FloatTensor (B, E)
                the conditional latent vectors
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor (B, L)
        """
        if len(self.prev_hs)==0: self.prev_hs.append(cdtnl)
        fx = self.features(x)
        fx = self.proj(fx) # (B, H)
        self.prev_hs.append(fx)
        encs = torch.stack(list(self.prev_hs), dim=1)
        encs = self.pos_enc(encs)
        encs = self.encoder( encs )
        # Only take the final prediction
        actn_encs = self.actn_encoder(encs)[:,-1]
        lang_encs = self.lang_encoder(encs)[:,-1]
        if self.lnorm:
            actn_encs = self.actnnorm(actn_encs)
            lang_encs = self.langnorm(lang_encs)
        self.hs = [ encs[:,-1], actn_encs[:,-1], lang_encs[:,-1] ]
        langs = []
        for dense in self.lang_denses:
            langs.append(dense(lang_encs))
        return self.output_fxn(self.actn_dense(actn_encs)), langs

    def forward(self, x, tasks, masks, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
            tasks: torch LongTensor (B,S)
                the index of the task for the particular data sequence.
                we only use the first element in the S direction. This
                means we are assuming the task index does not change
                over the course of the sequence.
            masks: torch LongTensor (B,S)
                Used to remove padding from the attention calculations.
                Ones denote locations that should be ignored. 0s denote
                locations that should be included.
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
            langs: torch FloatTensor (N,B,S,L)
        """
        cdtnl = self.cdtnl_lstm(self.cdtnl_idxs)
        seq_len = x.shape[1]
        b,s,c,h,w = x.shape
        fx = self.features(x.reshape(-1,c,h,w)).reshape(b*s,-1)
        fx = self.proj(fx).reshape(b,s,-1)
        # Add conditional vector
        fx = torch.cat([cdtnl[tasks[:,0]].unsqueeze(1), fx], dim=1)

        encs = self.pos_enc(fx)
        if masks is not None:
            masks = torch.cat([
                torch.zeros(b,1).to(self.get_device()),
                masks
            ], dim=1).bool() # cat for the conditional
        encs = self.encoder(
            encs,
            mask=self.fwd_mask[:s+1,:s+1],
            src_key_padding_mask=masks
        )
        actn_encs = self.actn_encoder(
            encs,
            self.fwd_mask[:s+1,:s+1],
            src_key_padding_mask=masks
        )
        lang_encs = self.lang_encoder(
            encs,
            self.fwd_mask[:s+1,:s+1],
            src_key_padding_mask=masks
        )
        # Remove first element because it's the conditional
        actn_encs = actn_encs[:,1:]
        lang_encs = lang_encs[:,1:]

        if self.lnorm:
            actn_encs = self.actnnorm(actn_encs)
            lang_encs = self.langnorm(lang_encs)

        actn_encs = actn_encs.reshape(b*s,-1)
        actns = self.actn_dense(actn_encs).reshape(b,s,-1)

        lang_encs = lang_encs.reshape(b*s,-1)
        langs = []
        for dense in self.lang_denses:
            langs.append(dense(lang_encs).reshape(b,s,-1))
        return self.output_fxn(actns), torch.stack(langs,dim=0)


class ConditionalLSTM(CoreModule):
    """
    This LSTM is used to process conditional sentences into a single
    latent vector.
    """
    def __init__(self, h_size, lang_size=None, *args,**kwargs):
        """
        h_size: int
            the hidden dimension size
        """
        super().__init__()
        self.h_size = h_size
        if lang_size is None: lang_size = CDTNL_LANG_SIZE
        self.embs = nn.Embedding(lang_size, self.h_size)
        self.lstm = nn.LSTMCell(self.h_size, self.h_size)
        self.layernorm_h = nn.LayerNorm(self.h_size)
        self.layernorm_c = nn.LayerNorm(self.h_size)
        self.h = None
        self.c = None

    def reset(self, batch_size):
        """
        Resets the recurrent state

        Args:
            batch_size: int
        """
        h = torch.zeros((batch_size, self.h_size))
        c = torch.zeros((batch_size, self.h_size))
        if self.is_cuda:
            h = h.to(self.get_device())
            c = c.to(self.get_device())
        return h,c

    def forward(self, x):
        """
        Processes the argued sequence to output a latent vector
        representation

        Args:
            x: torch LongTensor (B,S)
                a sequence of token indexes
        Returns:
            encs: torch FloatTensor (B,H)
                a batch of recurrent latent vectors
        """
        h,c = self.reset(len(x))
        for s in range(x.shape[1]):
            try:
                embs = self.embs(x[:,s])
                h,c = self.lstm(embs, (h, c))
                h = self.layernorm_h(h)
                c = self.layernorm_c(c)
            except:
                print("embs:", embs.get_device())
                print("h:", h.get_device())
                print("c:", c.get_device())
                assert False
        return h


