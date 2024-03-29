import collections
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import *
from cogmtc.utils.torch_modules import *
from cogmtc.utils.utils import update_shape, get_transformer_fwd_mask, max_one_hot, INEQUALITY, ENGLISH, PIRAHA, RANDOM, DUPLICATES, NUMERAL, BASELINE

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
    outsize = h_size if n_layers > 1 else outp_size
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

    Many models retain the h and c vectors of the most recent step
    in 2 lists called hs and cs. If the model has a language LSTM, the
    language lstm states are last in the list. Otherwise they progress
    smaller indices earlier in the network, larger indices later in
    the network.
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
        stack_context=True,
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
        lstm_lang=True,
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
        actn_fc_lnorm=None,
        lang_fc_lnorm=None,
        c_lnorm=True,
        lang_lnorm=False,
        fc_bnorm=False,
        stagger_preds=True,
        bottleneck=False,
        extra_lang_pred=False,
        legacy=False,
        targ_range=(1,17),
        rev_num=False,
        splt_feats=False,
        soft_attn=False,
        record_lang_stats=False,
        emb_ffn=False,
        one_hot_embs=False,
        lang_incl_layer=0,
        cut_lang_grad=False,
        incl_cdtnl=True,
        inpt_consol_emb_size=None,
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
            stack_context: bool
                if true, will take language embeddings and concatenate
                them along the h dimension to the visual latent vectors.
                only applies if incl_lang_inpt is true. Alternatively,
                vision and language will be alternated in context.
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
            targ_range: tuple of ints
                the training target range, (low, high) inclusive
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
                layer in the fully connected layers. Will override
                `actn_fc_lnorm` and `lang_fc_lnorm` only when `fc_lnorm`
                is true.
            actn_fc_lnorm: bool
                if true, adds a layernorm layer before each Linear
                layer in the action prediction fully connected layers
            lang_fc_lnorm: bool
                if true, adds a layernorm layer before each Linear
                layer in the language prediction fully connected layers
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
            rev_num: bool
                if true, the InptConsolidationModule will reverse the
                order of numerals up to the stop token when processesing
                language inputs. for example, the array [1,2,3,STOP]
                will become [3,2,1,STOP]
            splt_feats: bool
                effectively creates a separate convolutional network
                for the language pathway in the NSepLSTM variants.
                This ensures that the language and policy pathways do
                not overlap at all.
            soft_attn: bool
                if true, language inputs in sep models will be an
                attentional sum over the embeddings. Concretely, 
                the softmax outputs will act as attention (qk) which is
                applied over the embeddings (values). This creates a
                single vector as input as the sum of all possible
                embeddings.
            record_lang_stats: bool
                if true, will create a buffer variable that tracks
                the moving mean and variance of the language embeddings
                used as inputs into the policy network in models that
                use incl_lang_inpt.
            emb_ffn: bool
                if true, the embedding is processed through a
                feedforward network.
            one_hot_embs: bool
                if true, will maintain all embeddings as one-hot
                encodings.
            lang_incl_layer: int
                the lstm layer that the language predictions should be
                fed into in the model. 0 means the lang pred is fed
                into the policy as early as possible. -1 uses the
                last lstm in the policy chain.
            cut_lang_grad: bool
                if true, the gradient from the language pathway is not
                propagated beyond the first language lstm
            incl_cdtnl: bool
                if true, guarantees conditional vector is used.
                Otherwise the conditional is only used when there are
                more than 1 environment type
            inpt_consol_emb_size:
                if using word embeddings as language input, can specify
                the dimensionality of the embeddings
        """
        super().__init__()
        self.model_type = MODEL_TYPES.LSTM
        self.inpt_shape = inpt_shape
        self.actn_size = actn_size
        self.lang_size = lang_size
        self.h_size = h_size
        self.max_train_targ = targ_range[-1]
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
        """
            n_lstms: int
                the number of lstms for the model. Only applies for
                some model types.
        """
        self.n_lstms = 1
        self.env_types = env_types
        self.n_env_types = len(set(env_types))
        self.incl_cdtnl = incl_cdtnl
        self.inpt_consol_emb_size = inpt_consol_emb_size
        print("Including Conditional:", self.incl_cdtnl)
        self.env2idx = {k:i for i,k in enumerate(self.env_types)}
        self.n_envs = len(self.env_types)
        self.initialize_conditional_variables()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_vit_layers = n_vit_layers
        self.stack_context = stack_context
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
        self.max_ctx_len = 2*self.seq_len if max_ctx_len is None else\
                                                           max_ctx_len
        self.vision_type = vision_type
        self.learn_h = learn_h
        self.scaleshift = scaleshift
        self.fc_lnorm = fc_lnorm
        if self.fc_lnorm:
            self.actn_fc_lnorm = fc_lnorm
            self.lang_fc_lnorm = fc_lnorm
        else:
            self.actn_fc_lnorm = actn_fc_lnorm
            if actn_fc_lnorm is None: self.actn_fc_lnorm = self.fc_lnorm
            self.lang_fc_lnorm = lang_fc_lnorm
            if lang_fc_lnorm is None: self.lang_fc_lnorm = self.fc_lnorm
        self.c_lnorm = c_lnorm
        self.lang_lnorm = lang_lnorm
        self.fc_bnorm = fc_bnorm
        self.stagger_preds = stagger_preds
        self.bottleneck = bottleneck
        self.extra_lang_pred = extra_lang_pred
        self.legacy = legacy
        self.rev_num = rev_num
        self.splt_feats = splt_feats
        self.soft_attn = soft_attn
        self.record_lang_stats = record_lang_stats
        self.emb_ffn = emb_ffn
        self.one_hot_embs = one_hot_embs
        self.lang_incl_layer = lang_incl_layer
        self.cut_lang_grad = cut_lang_grad

    def initialize_conditional_variables(self):
        """
        Creates the conditional lstm, the conditional long indices, and
        the conditional batch distribution tensor. The cdtnl_batch
        tensor is a way to use the same conditional for all appropriate
        batch rows at the same time. At training time, we use 
        `repeat_interleave` to expand cdtnl_batch appropriately.
        """
        if "gordongames-v11" in self.env_types or "gordongames-v12" in\
                                                        self.env_types:
            lang_size = CDTNL_LANG_SIZE + self.max_train_targ
        else: lang_size = CDTNL_LANG_SIZE 
        self.cdtnl_lstm = ConditionalLSTM(
            self.h_size, lang_size=lang_size
        )
        max_len = max([len(v) for v in TORCH_CONDITIONALS.values()])
        cdtnl_idxs = torch.zeros(len(self.env_types),max_len).long()
        add_n2cdtnls = []
        for env_type in self.env_types:
            k = self.env2idx[env_type]
            l = len(TORCH_CONDITIONALS[env_type])
            cdtnl_idxs[k,:l] = TORCH_CONDITIONALS[env_type]
            if "gordongames-v11"==env_type or"gordongames-v12"==env_type:
                add_n2cdtnls.append(k)
        self.register_buffer("cdtnl_idxs", cdtnl_idxs)
        if len(add_n2cdtnls)>0:
            self.register_buffer("add_n2cdtnls",
                torch.LongTensor(add_n2cdtnls)
            )
        else:
            self.add_n2cdtnls = None

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
            lnorm=self.actn_fc_lnorm,
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
                    fc_lnorm=self.lang_fc_lnorm,
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
                    lnorm=self.lang_fc_lnorm,
                    scaleshift=self.scaleshift,
                    legacy=self.legacy
                )
            self.lang_denses.append(dense)

    def process_lang_preds(self, langs):
        """
        Assists in recording the language prediction that will
        potentially be used in the next time step. Averages over the
        preds, reshapes them and then takes the argmax

        Args:
            langs: list of torch FloatTensors [(B,L), (B,L), ...]
        Returns:
            lang: torch LongTensor (B,M) or FloatTensor (B, L)
                the average over the langs list then the argmax over L
                or L//max_char_seq if using NUMERAL variants. Will
                return softmax over L dimension if using soft_attn
        """
        lang = langs[0].detach().data/len(langs)
        for i in range(1,len(langs)):
            lang = lang + langs[i].detach().data/len(langs)
        if self.soft_attn:
            lang = torch.softmax(
                lang.reshape(len(self.lang),self.max_char_seq,-1),dim=-1
            )
        else:
            lang = torch.argmax(
                lang.reshape(len(self.lang),self.max_char_seq,-1),dim=-1
            ).long()
        return lang

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
                       h_size,
                       use_count_words=None,
                       max_char_seq=1,
                       STOP=1,
                       null_idx=0,
                       drop_p=0,
                       rev_num=False,
                       soft_attn=False,
                       record_lang_stats=False,
                       alpha=0.99,
                       emb_ffn=False,
                       one_hot_embs=False,
                       emb_size=None,
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
            emb_size: int
                the embedding size.
            max_char_seq: int
            STOP: int
                index of stop token (if one exists). only matters if
                max_char_seq is greater than 1
            null_idx: int
                index of NULL token (if one exists). only matters if
                max_char_seq is greater than 1
            drop_p: float
            rev_num: bool
                if true, will reverse the order of numerals in the
                last dimension up to the stop token when processing
                the language inputs.
            soft_attn: bool
                if true, language inputs to LSTM will be an
                attentional sum over the embeddings. In this module,
                this means that instead of receiving token indices,
                it will receive a softmax over the potential embeddings. 
                This module will use the softmax as attention (qk) which
                is applied over the embeddings (values). This creates a
                single vector as input as the sum of all possible
                embeddings.
            record_lang_stats: bool
                if true, will create a buffer variable that tracks
                the moving mean and variance of the language embeddings
                used as inputs into the policy network in models that
                use incl_lang_inpt.
            alpha: float [0,1]
                the moving average factor if record_lang_stats is true
            emb_ffn: bool
                if true, the embedding is processed through a
                feedforward network.
            one_hot_embs: bool
                if true, will maintain all embeddings as one-hot
                encodings.
        """
        super().__init__()
        self.lang_size = lang_size
        self.h_size = h_size
        self.one_hot_embs = one_hot_embs
        self.emb_size = h_size if emb_size is None else emb_size
        self.use_count_words = use_count_words
        self.mcs = 1 if max_char_seq is None or max_char_seq < 1\
                     else max_char_seq
        self.STOP = STOP
        self.null_idx = null_idx
        self.drop_p = drop_p
        self.rev_num = rev_num
        self.soft_attn = soft_attn
        self.record_lang_stats = record_lang_stats
        self.alpha = alpha
        self.emb_ffn = emb_ffn

        self.embeddings = nn.Embedding(self.lang_size,self.emb_size)
        if self.one_hot_embs:
            self.embeddings.weight.data = torch.eye(self.lang_size)
            self.embeddings.weight.requires_grad = False
            self.emb_size = self.lang_size
        if self.record_lang_stats:
            self.register_buffer("emb_mean", torch.zeros(1,self.emb_size))
            self.register_buffer("emb_std", torch.ones(1,self.emb_size))
        self.dropout = nn.Dropout(p=self.drop_p)
        if self.use_count_words == NUMERAL:
            self.lstm_consol = ContainedLSTM( self.emb_size, self.emb_size )
        if self.emb_size==self.h_size:
            self.consolidator = nn.Sequential(
                nn.Linear(self.h_size, self.h_size),
                nn.LayerNorm(self.h_size),
                nn.ReLU()
            )
            self.proj = nn.Linear(self.h_size, self.h_size)
        else:
            if self.emb_ffn:
                self.consolidator = nn.Sequential(
                    nn.Linear(self.emb_size, self.h_size),
                    nn.LayerNorm(self.h_size),
                    nn.ReLU()
                )
                self.proj = nn.Linear(self.h_size, self.h_size)
            else:
                self.proj = nn.Linear(self.emb_size, self.h_size)

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

    @staticmethod
    def reverse_numerals(x, stop_token):
        """
        Reverses the numerals in an array while keeping the stop
        token in the same location. if no stop token exists in the
        sequence, assumes last location is stop token.
    
        start: [1,2,3,STOP,a1,a2,...]
        end:   [3,2,1,STOP,a1,a2,...]
    
        Args:
            x: torch Long tensor (..., S)
                the tensor to be reversed
            stop_token: int
                the token indicating the stop location
        Returns:
            rev: torch Long Tensor (..., S)
                the reversed tensor
        """
        device = x.get_device()
        og_shape = x.shape
        x = x.clone().reshape(-1,x.shape[-1])
        x[:,-1] = stop_token
        stop_idxs = torch.argmax( (x==stop_token).float(), dim=-1 )
        for i in range(1,x.shape[-1]):
            rows = (stop_idxs==(i+1))
            temp = x[rows]
            if len(temp) > 0:
                rev = torch.arange(i,-1,-1,device=device).long()
                fwd = torch.arange(x.shape[-1], device=device).long()
                fwd[:len(rev)] = rev
                x[rows] = torch.index_select(temp, dim=-1, index=fwd)
        return x.reshape(og_shape)

    def forward(self, x, avg_embs=False, sample_embs=False):
        """
        Either performs an attention operation over embeddings (if
        soft_attn is true), or selects embeddings, or selects embeddings
        and combines them (if NUMERAL).

        Args:
            x: torch LongTensor (B, S) or torch FloatTensor (B,S,L)
                a sequence of indices or softmax values over language
                embeddings.
            avg_embs: bool
                if true, will take the average of all embeddings rather
                than the embeddings themselves. This is used as a
                comparison to English speakers without language.
            sample_embs: bool
                if true and self.record_lang_stats is true, will sample
                embeddings from the recorded mean and std
        Returns:
            fx: torch tensor (B, H)
        """
        if avg_embs and len(x.shape)==2:
            b,s = x.shape
            if self.record_lang_stats:
                embs = self.emb_mean.repeat((b,1))
            else:
                embs = self.embeddings.weight.mean(0)[None].repeat((b,1))
        elif self.soft_attn and x.dtype == torch.FloatTensor().dtype:
            if self.use_count_words == NUMERAL: raise NotImplemented
            # Attention over embeddings
            embs = torch.matmul(x, self.embeddings.weight)
            if self.record_lang_stats:
                a = self.alpha
                e = embs.reshape(1,embs.shape[-1])
                self.emb_mean = a*self.emb_mean + (1-a)*e.mean(0)
                self.emb_std = a*self.emb_std + (1-a)*e.std(0)
        else:
            x = x.clone()
            x[x<0] = self.null_idx
            embs = self.embeddings(x)
            if self.record_lang_stats:
                a = self.alpha
                e = embs.reshape(1,embs.shape[-1])
                self.emb_mean = a*self.emb_mean + (1-a)*e.mean(0)
                self.emb_std = a*self.emb_std + (1-a)*e.std(0)
        embs = self.dropout(embs)
        if self.use_count_words == NUMERAL:
            if self.rev_num:
                x = InptConsolidationModule.reverse_numerals(x,self.STOP)
            mask = self.get_mask(x, self.STOP, incl_token=True)
            embs = self.lstm_consol(embs, mask)

        # If embs shape is (B,1,E)
        if len(embs.shape)!=2 and embs.shape[1]==1:
            embs = embs[:,0]
        if self.emb_ffn:
            #I think we can remove this if statement, but I'm not certain
            if len(embs.shape)==2:
                embs = self.consolidator(embs)
            return self.proj(embs)
        elif self.h_size!=self.emb_size:
            return self.proj(embs)
        return embs


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

class SimpleFFN(Model):
    """
    Use this module to feed the visual input into a feed forward network
    """
    def __init__(self, feats_only=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = MODEL_TYPES.CNN
        img_size = int(np.prod(self.inpt_shape[-3:]))
        self.features = nn.Sequential(
            Flatten(),
            get_fcnet(
                inpt_size=img_size,
                h_size=self.h_mult*self.h_size,
                outp_size=self.h_size,
                n_layers=len(self.depths),
                drop_p=self.feat_drop_p,
                bnorm=self.fc_bnorm,
                lnorm=self.fc_lnorm,
                scaleshift=self.scaleshift,
                actv_fxn=self.actv_fxn,
            )
        )
        self.flat_size = self.h_size
        self.shapes = [ *self.inpt_shape[-3:] ]

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
        fx = self.features(x.reshape(len(x), -1))
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
        adders = [0 for _ in self.depths]
        groups = 1
        if self.splt_feats:
            for i in range(1,len(self.depths)):
                adders[i] = self.depths[i]
            groups = 2
        for i in range(len(self.depths)-1):
            # CONV
            modules.append(
                nn.Conv2d(
                    self.depths[i] + adders[i],
                    self.depths[i+1] + adders[i+1],
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=self.paddings[i],
                    groups=max(int(groups*(i>0)), 1)
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
                "rev_num": self.rev_num,
                "soft_attn": self.soft_attn,
                "record_lang_stats": self.record_lang_stats,
                "emb_ffn": self.emb_ffn,
                "one_hot_embs": self.one_hot_embs,
                "emb_size": self.inpt_consol_emb_size,
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

    def forward(self,x,dones,tasks,masks,lang_inpts=None,
                                         n_targs=None,
                                         *args,**kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
            dones: torch Long Tensor (B, S)
                the done signals for the environment. the h and c
                vectors are reset when encountering a done signal
            tasks: torch Long Tensor (B, S)
                the task signal corresponding to each environment. This
                is an integer that corresponds to the task index in
                self.env2idx
            masks: torch LongTensor or BoolTensor (B,S)
                Used to avoid continuing calculations in the sequence
                when the sequence is over. Ones in the mask denote
                that the time step should be ignored, zeros should
                be included.
            lang_inpts: None or LongTensor (B,S,M)
                if not None and self.incl_lang_inpt is true, lang_inpts
                are used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
                Note, these are the correct labels for the timestep.
                So, if you are predicting language, you will not want
                to use the corresponding timestep from this sequence
                as input to that prediction.
            n_targs: None or LongTensor (B,S)
                only applies for gordongames-v11 and v12. this is a
                vector indicating the number of target objects for the
                episode.
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
        give_n = self.add_n2cdtnls is not None
        # Give n tasks are tasks in which the target quantity for the
        # episode is given as a part of the conditional rather than
        # the visual input.
        if give_n:
            # Shift and clip n_targs for correct embedding selection
            n_targs = n_targs + CDTNL_LANG_SIZE - 1# -1 because no 0 targ
            max_val = CDTNL_LANG_SIZE+self.max_train_targ
            n_targs[n_targs>=max_val] = max_val-1

            n_targ_embs = self.cdtnl_lstm.embs(n_targs)

            # zero embs for the not give n type tasks
            task_mask = torch.isin(tasks, self.add_n2cdtnls).float()
            n_targ_embs = n_targ_embs*task_mask[...,None]

        for s in range(seq_len):
            if masks[:,s].sum() == len(masks):
                actns.append(torch.zeros_like(actns[-1]))
                langs.append(torch.zeros_like(langs[-1]))
            else:
                lang_inpt=None if lang_inpts is None else lang_inpts[:,s]
                cdt = cdtnl[tasks[:,s]]
                # n_targs is 0 if not a give n type task
                if give_n: cdt = cdt + n_targ_embs[:,s]
                actn, lang = self.step(
                  x[:,s], cdt, masks[:,s], lang_inpt
                )
                actns.append(actn.unsqueeze(1))
                if self.n_lang_denses == 1 and not self.extra_lang_pred:
                    lang = lang[0].unsqueeze(0).unsqueeze(2) # (1, B, 1, L)
                else:
                    lang = torch.stack(lang, dim=0).unsqueeze(2)# (N, B, 1, L)
                langs.append(lang)
            self.hs, self.cs, self.lang = self.partial_reset(dones[:,s])
            self.prev_hs.append([h.detach().data for h in self.hs])
            self.prev_cs.append([c.detach().data for c in self.cs])
            self.prev_langs.append(self.lang)
        return ( torch.cat(actns, dim=1), torch.cat(langs, dim=2) )

class SeparateLSTM(LSTMOffshoot):
    """
    A model with three LSTMs total. One for language that operates
    independently. Then two sequential LSTMs for the actions that
    receives a cut-gradient language input.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def step(self, x, cdtnl, mask=None,lang_inpt=None,blank_lang=False,
                                                      avg_lang=False,
                                                      *args,**kwargs):
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
            blank_lang: bool
                if true, blanks out the language before inputting
                into the model. only applies if incl_lang_inpt is
                true
            avg_lang: bool
                if true, uses the average of the embeddings as the
                lang inpts. only applies if incl_lang_inpt is true
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
        lang_inpt = self.lang_consolidator( lang_inpt,avg_embs=avg_lang )
        if blank_lang: lang_inpt = torch.zeros_like(lang_inpt)

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


class NSepLSTM(SeparateLSTM):
    """
    This is a generalization of the SeparateLSTM. n_lstms determines how
    many LSTMs are used for the policy network. In all cases, only one
    LSTM is used for language generation. The language output is used
    to select an embedding that is fed into the first LSTM in the policy
    network. No gradients are backpropagated into the language lstm from
    the policy network.

    The hs and cs members are lists of hidden states in which the
    language lstm states are last in the list.
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
        if n_lstms < 2: print("NSepLSTM always has at least 2 lstms")
        self.n_lstms = n_lstms # The number of LSTMs including the lang lstm 
        self.n_lang_denses = 1
        del self.actn_lstm
        del self.lstm

        size = self.flat_size + self.h_size # add hsize for condtional
        self.lang_lstm = nn.LSTMCell(size, self.h_size)

        lil = self.lang_incl_layer
        if lil == 0:
            size = size + self.h_size # Will include lang prediction here
        self.lstms = nn.ModuleList([
            nn.LSTMCell(size, self.h_size)
        ])

        if self.bottleneck:
            assert lil == 0
            self.lstms[0] = nn.LSTMCell(self.h_size, self.h_size)

        for i in range(1, self.n_lstms-1):
            size = self.h_size
            if self.skip_lstm:
                assert not self.bottleneck,\
                            "bottleneck and skip_lstm are incompatible"
                # add additional h_size here for conditional input
                size = self.h_size + size + self.flat_size
            if lil==i or (lil==-1 and i==self.n_lstms-2):
                size += self.h_size
            self.lstms.append( nn.LSTMCell(size, self.h_size) )

        if self.learn_h:
            self.get_inits()
        self.reset(1)

        self.make_actn_dense()
        self.make_lang_denses()

    def step(self,x,cdtnl,mask=None,lang_inpt=None,blank_lang=False,
                                                   avg_lang=False,
                                                   *args,**kwargs):
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
            lang_inpt: None or LongTensor (B,M) or (B,)
                if not None and self.incl_lang_inpt is true, lang_inpt
                is used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
            blank_lang: bool
                if true, blanks out the language before inputting
                into the model. only applies if incl_lang_inpt is
                true
            avg_lang: bool
                if true, uses the average of the language embeddings
                as the input to the policy. only applies if
                incl_lang_inpt is true
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor [ (B, L) ]
        """
        # Not that the mask gets flipped to where 1 means keep the
        # calculation in the data
        if mask is None:
            mask = torch.ones(x.shape[0]).bool()
        else:
            mask = (1-mask).bool()
        device = self.get_device()
        self.lang = self.lang.to(device)
        mask = mask.to(device)
        for i in range(self.n_lstms):
            self.hs[i] = self.hs[i].to(device)
            self.cs[i] = self.cs[i].to(device)

        fx = self.features(x[mask])
        lang_fx = fx
        if self.splt_feats:
            midpt = int(fx.shape[1]//2)
            lang_fx = fx[:,:midpt]
            fx = fx[:,midpt:]
        lang_fx = lang_fx.reshape(len(lang_fx),-1)
        fx = fx.reshape(len(fx), -1) # (B, N)
        inpt = [lang_fx, cdtnl[mask]]
        cat = torch.cat(inpt, dim=-1)

        h,c = (self.hs[-1][mask], self.cs[-1][mask])
        if self.lang_lnorm:
            h = self.h_lang_lnorm(h)
            if self.c_lnorm:
                c = self.c_lang_lnorm(c)
        if self.cut_lang_grad:
            lang_h, lang_c = self.lang_lstm( cat.data, (h,c) )
        else:
            lang_h, lang_c = self.lang_lstm( cat, (h,c) )
        lang = self.lang_denses[0](lang_h)

        if lang_inpt is None:
            lang_inpt = self.process_lang_preds([lang])
        else:
            lang_inpt = lang_inpt[mask]
        lang_inpt = self.lang_consolidator(lang_inpt,avg_embs=avg_lang)
        if blank_lang: lang_inpt = torch.zeros_like(lang_inpt)

        if self.bottleneck:
            cat = lang_inpt
        elif self.lang_incl_layer==0:
            inpt = [fx, cdtnl[mask], lang_inpt]
            cat = torch.cat(inpt, dim=-1)
        else:
            inpt = [fx, cdtnl[mask]]
            cat = torch.cat(inpt, dim=-1)

        h, c = self.lstms[0]( cat, (self.hs[0][mask], self.cs[0][mask]) )
        if self.lnorm:
            h = self.layernorm_h(h)
            if self.c_lnorm:
                c = self.layernorm_c(c)
        hs = [h]
        cs = [c]

        lil = self.lang_incl_layer
        for i in range(1, len(self.lstms)):
            inpt = h
            if lil==i or (lil==-1 and i==len(self.lstms)-1):
                inpt = [inpt, lang_inpt]
                inpt = torch.cat(inpt, dim=-1)
            elif self.skip_lstm: 
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


class PreNSepLSTM(LSTMOffshoot):
    """
    This model builds off the NSepLSTM model type. This one is different
    in that allows you to specify the n_pre_lstms
    and n_lang_lstms counts. These allow for an arbitrary number of
    lstms following the visual vector before the language split, and
    an arbitrary number of lstms for the language pathway.

    We apply layer norm after the lstms, not before.
    """
    def __init__(self, n_pre_lstms=0, n_lang_lstms=1, n_actn_lstms=2,
                                                      *args, **kwargs):
        """
        Args:
            n_pre_lstms: int
                determines the number of lstms directly following the
                visual latent vector
            n_lang_lstms: int
                determines the number of LSTMs in the language pathway
                following the pre pathway
            n_actn_lstms: int
                determines the number of LSTMs in the actn pathway
                following the pre pathway
        """
        super().__init__(*args, **kwargs)
        if self.bottleneck: raise NotImplemented
        if self.skip_lstm: raise NotImplemented
        if self.c_lnorm: print("C Lnorm is stupid. Ignoring it")
        self.n_pre = n_pre_lstms
        # Do not want shared lstm pathway if completely splitting visual
        # pathway
        assert not (self.splt_feats and self.n_pre>0) 
        self.n_lang = n_lang_lstms
        self.n_actn = n_actn_lstms
        self.n_lstms = self.n_pre + self.n_lang + self.n_actn
        self.n_lang_denses = 1

        # Pre LSTMS
        self.pre_lstms = nn.ModuleList([
          nn.LSTMCell(self.h_size,self.h_size)for i in range(self.n_pre)
        ])

        # Lang LSTMS
        self.lang_lstms = nn.ModuleList([
          nn.LSTMCell(self.h_size,self.h_size)for i in range(self.n_lang)
        ])

        # Actn LSTMS
        size = self.h_size
        if self.incl_lang_inpt:
            if self.one_hot_embs:
                size += self.lang_size
            else: size += self.h_size
        self.lstms = nn.ModuleList([ nn.LSTMCell(size, self.h_size) ])
        for i in range(1, self.n_actn):
            self.lstms.append( nn.LSTMCell(self.h_size, self.h_size) )

        if self.learn_h: self.get_inits()
        self.reset(1)

        if self.lnorm:
            self.pre_lnorms = nn.ModuleList(
                [nn.LayerNorm(self.h_size) for _ in range(self.n_pre)]
            )
            self.actn_lnorms = nn.ModuleList(
                [nn.LayerNorm(self.h_size) for _ in range(self.n_actn)]
            )
        if self.lang_lnorm:
            self.lang_lnorms = nn.ModuleList(
                [nn.LayerNorm(self.h_size) for _ in range(self.n_lang)]
            )

        self.make_actn_dense()
        self.make_lang_denses()
        size = self.flat_size + self.h_size*self.incl_cdtnl
        self.cdtnl_proj = nn.Linear(
            size,self.h_size
        )
        if self.splt_feats:
            self.lang_cdtnl_proj = nn.Linear(
                size,self.h_size
            )

    def step(self,x,cdtnl,mask=None,lang_inpt=None,blank_lang=False,
                                                   avg_lang=False,
                                                   *args,**kwargs):
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
            lang_inpt: None or LongTensor (B,M) or (B,)
                if not None and self.incl_lang_inpt is true, lang_inpt
                is used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq
            blank_lang: bool or tensor
                if True is argued, will replace input language embeddings
                with zeros before inputting into model.
                if tensor is argued, replaces the input language
                embeddings with the argued tensor before inputting
                into the model. only applies if incl_lang_inpt is
                true
            avg_lang: bool
                if true, uses the average of the language embeddings
                as the input to the policy. only applies if
                incl_lang_inpt is true
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor [ (B, L) ]
        """
        # Not that the mask gets flipped to where 1 means keep the
        # calculation in the data
        if mask is None:
            mask = torch.ones(x.shape[0]).bool()
        else:
            mask = (1-mask).bool()
        device = self.get_device()
        self.lang = self.lang.to(device)
        mask = mask.to(device)
        for i in range(self.n_lstms):
            self.hs[i] = self.hs[i].to(device)
            self.cs[i] = self.cs[i].to(device)

        # Features
        fx = self.features(x[mask])
        if self.splt_feats:
            midpt = int(fx.shape[1]//2)
            fx = fx[:,midpt:]
            lang_fx = fx[:,:midpt]
            lang_fx = lang_fx.reshape(len(lang_fx),-1)
            cat = lang_fx
            if self.incl_cdtnl:
                cat = torch.cat([lang_fx, cdtnl[mask]], dim=-1)
            lang_h = self.lang_cdtnl_proj(cat)
        fx = fx.reshape(len(fx), -1) # (B, N)
        cat = fx
        if self.incl_cdtnl:
            cat = torch.cat([fx, cdtnl[mask]], dim=-1)
        fx = self.cdtnl_proj(cat)

        # Pre Pathway
        hs = []
        cs = []
        if self.n_pre > 0:
            h = fx
            for i in range(self.n_pre):
                h, c = self.pre_lstms[i](
                    h, (self.hs[i][mask], self.cs[i][mask])
                )
                if self.lnorm: h = self.pre_lnorms[i](h)
                hs.append(h)
                cs.append(c)
            fx = h

        if not self.splt_feats:
            lang_h = fx
        h = fx

        # Lang Pathway
        lang_hs = []
        lang_cs = []
        for i in reversed(range(self.n_lang)):
            idx = self.n_lang-1-i
            if self.cut_lang_grad and i==self.n_lang-1:
                lang_h, lang_c = self.lang_lstms[idx](
                    lang_h.data, (self.hs[-i][mask], self.cs[-i][mask])
                )
            else:
                lang_h, lang_c = self.lang_lstms[idx](
                    lang_h, (self.hs[-i][mask], self.cs[-i][mask])
                )
            if self.lang_lnorm: lang_h = self.lang_lnorms[idx](lang_h)
            lang_hs.append(lang_h)
            lang_cs.append(lang_c)

        # Lang Prediction and Manipulation
        lang = self.lang_denses[0](lang_h)

        if self.incl_lang_inpt:
            if lang_inpt is None:
                lang_inpt = self.process_lang_preds([lang])
            else:
                lang_inpt = lang_inpt[mask]
            lang_inpt = self.lang_consolidator(
                lang_inpt, avg_embs=avg_lang
            )
            if type(blank_lang)!=type(bool()):
                lang_inpt[...,:] = blank_lang
            elif blank_lang:
                lang_inpt = torch.zeros_like(lang_inpt)
            h = [h, lang_inpt]
            h = torch.cat(h, dim=-1)

        # Action Pathway
        for i in range(self.n_actn):
            # Keep all lstm type h and c vectors in single list. Thus
            # we need to be careful with their ordering.
            idx = i+self.n_pre
            h, c = self.lstms[i](
                h, (self.hs[idx][mask], self.cs[idx][mask])
            )
            if self.lnorm: h = self.actn_lnorms[i](h)
            hs.append(h)
            cs.append(c)

        # Actn Prediction
        actn = self.actn_dense(hs[-1])

        # Cleanup
        for lang_h, lang_c in zip(lang_hs, lang_cs):
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
        temp_actn[mask] = actn
        temp_lang = torch.zeros(x.shape[0],lang.shape[-1],device=device)
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

    def step(self, x, cdtnl, lang_inpt=None, blank_lang=False,
                                             avg_lang=False,
                                            *args, **kwargs):
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
            blank_lang: bool
                if true, zeros out the lang inputs. only applies if
                incl_lang_inpt is true
            avg_lang: bool
                if true, uses the average of the embeddings as the
                lang inpts. only applies if incl_lang_inpt is true
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
            lang_inpt=self.lang_consolidator(lang_inpt,avg_embs=avg_lang)
            if blank_lang: lang_inpt = torch.zeros_like(lang_inpt)
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

        self.lstms = nn.ModuleList([self.lstm])
        if self.lnorm:
            self.h_lnorms = nn.ModuleList([self.layernorm_h])
            self.c_lnorms = nn.ModuleList([self.layernorm_c])

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

    def step(self, x, cdtnl, lang_inpt=None, blank_lang=False,
                                             avg_lang=False,
                                             *args, **kwargs):
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
            blank_lang: bool
                if true, zeros out the language before inputting
                into the model. only applies if incl_lang_inpt is
                true
            avg_lang: bool
                if true, uses the average of the embeddings as the
                lang inpts. only applies if incl_lang_inpt is true
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
        if self.max_char_seq is None: self.max_char_seq = 1
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

        # The consolidators are either the identity function or they
        # are a class that assists with processing the language
        # for the next timestep
        self.lang_consolidator = identity

        # Linear Projection
        self.flat_size = self.cnn.flat_size
        size = self.h_size
        self.lang_consolidator = identity
        if self.incl_lang_inpt:
            if self.stack_context:
                size = self.h_size//2
            consolidator_kwargs = {
                "lang_size": self.lang_size,
                "h_size": size,
                "max_char_seq": self.max_char_seq,
                "STOP": self.STOP,
                "drop_p": self.lang_inpt_drop_p,
                "use_count_words": self.use_count_words,
                "rev_num": self.rev_num,
                "soft_attn": self.soft_attn,
                "record_lang_stats": self.record_lang_stats,
                "emb_ffn": self.emb_ffn,
                "one_hot_embs": self.one_hot_embs,
                "emb_size": self.inpt_consol_emb_size,
            }
            self.lang_consolidator = InptConsolidationModule(
                **consolidator_kwargs
            )
        self.proj = nn.Sequential(
            Flatten(),
            nn.Linear(self.flat_size, size)
        )

        # Transformer
        self.pos_enc = PositionalEncoding(
            self.h_size,
            self.feat_drop_p
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.h_size,
            nhead=self.n_heads,
            dim_feedforward=self.h_mult*self.h_size,
            dropout=self.feat_drop_p,
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

    def step(self, x, cdtnl, lang_inpt=None, blank_lang=False,
                                             *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
            cdtnl: torch FloatTensor (B, E)
                the conditional latent vectors
            lang_inpt: None or torch LongTensor (B, M)
                a sequence of language indicies that will be consolidated
                and used as input to the lstm. for non NUMERAL models,
                M will be equal to 1.
            blank_lang: bool
                if true, blanks out the language before inputting
                into the model. only applies if incl_lang_inpt is
                true
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor [ (B, L) ]
        """
        if len(self.prev_hs)==0: self.prev_hs.append(cdtnl)
        fx = self.features(x)
        fx = self.proj(fx) # (B, H)
        if self.use_count_words!=BASELINE and self.incl_lang_inpt:
            # We dont want to use lang_inpt here because it's the
            # ground truth for this step. So, we store it in self.lang
            # for the next step. This only applies if teacher_force_val
            # is true.
            if self.lang is not None and lang_inpt is None:
                lang = self.process_lang_preds(self.lang)
            if blank_lang or self.lang is None:
                lang = torch.zeros_like(fx)
            else:
                lang = self.lang_consolidator(lang)
            # concat vision and language vectors along h
            if self.stack_context:
                fx = torch.cat([fx,lang], axis=-1)
            # alternate vision and language vectors in context
            else:
                if not self.lstm_lang_first:
                    fx,lang = lang,fx
                self.prev_hs.append(lang)
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
        # Need to use lang_inpt for next step, not this step
        if lang_inpt is None:
            self.lang = langs
        else:
            self.lang = lang_inpt
        return self.output_fxn(self.actn_dense(encs)), langs

    def forward(self, x, tasks, masks=None, lang_inpts=None,
                                            n_targs=None,
                                            blank_lang=False,
                                            *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
            tasks: torch LongTensor (B,S)
            masks: torch LongTensor (B,S)
                Used to remove padding from the attention calculations.
                Ones denote locations that should be ignored. 0s denote
                locations that should be included.
            lang_inpts: None or LongTensor (B,S,M)
                if not None and self.incl_lang_inpt is true, lang_inpts
                are used as an additional input into the lstm. They
                should be token indicies. M is the max_char_seq.
                Note, these are the correct labels for the timestep.
                So, if you are predicting language, you will not want
                to use the corresponding timestep from this sequence
                as input to that prediction.
            n_targs: None or LongTensor (B,S)
                only applies for gordongames-v11 and v12. this is a
                vector indicating the number of target objects for the
                episode.
            blank_lang: bool
                if true, blanks out the language before inputting
                into the model. only applies if incl_lang_inpt is
                true
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
        inpt = [cdtnl[tasks[:,0]].unsqueeze(1)]

        # Handle Language. Always use teacher forcing during training
        if self.use_count_words!=BASELINE and self.incl_lang_inpt:
            if lang_inpts is None or blank_lang:
                lang = torch.zeros_like(fx)
            else:
                lang = self.lang_consolidator(lang_inpts)

            # Prepend zeros and remove final language token to avoid
            # providing ground truth to transformer. Remember that
            # torch pad takes a backwards tuple because they're dumb af
            # In this case, lang needs to have 3 dims where seq is the
            # middle
            lang = torch.nn.functional.pad(lang[:,:-1],(0,0,1,0))
            # concat vision and language vectors along h
            if self.stack_context:
                fx = torch.cat([fx,lang], axis=-1)
            # alternate vision and language vectors in context
            else:
                # Need to manipulate mask to accommodate expansion of
                # sequence length
                if masks is not None: 
                    temp = torch.zeros(
                        (b,2*masks.shape[1]),
                        dtype=masks.dtype,
                        device=fx.get_device()
                    )
                    temp[:,::2] = masks
                    temp[:,1::2] = masks
                    masks = temp
                temp = torch.zeros(
                    (b,2*s,fx.shape[-1]),
                    device=fx.get_device()
                )
                if not self.lstm_lang_first:
                    fx,lang = lang,fx
                temp[:,::2] = lang
                temp[:,1::2] = fx
                fx = temp
        inpt.append(fx)

        fx = torch.cat(inpt, dim=1)
        encs = self.pos_enc(fx)
        if masks is not None:
            masks = torch.nn.functional.pad(masks, (1,0)).bool()
        encs = self.encoder(
            encs,
            mask=self.fwd_mask[:encs.shape[1],:encs.shape[1]],
            src_key_padding_mask=masks
        )
        encs = encs[:,1:] # Remove conditional vector
        if encs.shape[1] > s+1:
            encs = encs[:,1::2]
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
            self.h_mult*self.h_size,
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

