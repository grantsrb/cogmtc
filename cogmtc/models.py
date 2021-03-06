import collections
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import *
from cogmtc.utils.torch_modules import *
from cogmtc.utils.utils import update_shape, get_transformer_fwd_mask
from cogmtc.envs import TORCH_CONDITIONALS, CDTNL_LANG_SIZE
import matplotlib.pyplot as plt

def get_fcnet(inpt_size,
              outp_size,
              n_layers=2,
              h_size=256,
              noise=0,
              drop_p=0,
              bnorm=False,
              lnorm=False,
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
    """
    outsize= h_size if n_layers > 1 else outp_size
    block = [ nn.Linear(inpt_size, outsize) ]
    prev_size = outsize
    for i in range(1, n_layers):
        block.append( GaussianNoise(noise) )
        if bnorm: block.append( nn.BatchNorm1d(outsize) )
        if lnorm: block.append( nn.LayerNorm(outsize) )
        block.append( nn.Dropout(drop_p) )
        block.append( globals()[actv_fxn]() )
        block.append( ScaleShift((outsize,)) )
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
            return next(self.parameters()).get_device()
        except:
            return False

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
        lstm_lang_first=True,
        env_types=["gordongames-v4"],
        n_heads=8,
        n_layers=3,
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
        lstm_lang=False,
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
            lstm_lang: bool
                if you want to use an additional lstm to output the
                language for numeral systems, set this to true. if false
                and using a numeral system, a single dense net makes
                all numeral predictions at the same time. Does not
                affect anything if not using numeral system
        """
        super().__init__()
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
        self.n_lang_denses = n_lang_denses
        self._trn_whls = nn.Parameter(torch.ones(1), requires_grad=False)
        self.lstm_lang_first = lstm_lang_first
        self.n_lstms = 1
        self.env_types = env_types
        self.n_envs = len(self.env_types)
        self.initialize_conditional_variables()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.max_char_seq = max_char_seq
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

    def initialize_conditional_variables(self):
        """
        Creates the conditional lstm, the conditional long indices, and
        the conditional batch distribution tensor. The cdtnl_batch
        tensor is a way to use the same conditional for all appropriate
        batch rows at the same time. At training time, we use 
        `repeat_interleave` to expand cdtnl_batch appropriately.
        """
        self.cdtnl_lstm = ConditionalLSTM(self.h_size)
        max_len = max([len(v) for v in TORCH_CONDITIONALS.values()])
        cdtnl_idxs = torch.zeros(len(self.env_types),max_len).long()
        cdtnl_batch = torch.arange(self.n_envs).long()
        for i,env_type in enumerate(self.env_types):
            l = len(TORCH_CONDITIONALS[env_type])
            cdtnl_idxs[i,:l] = TORCH_CONDITIONALS[env_type]
        self.register_buffer("cdtnl_batch", cdtnl_batch)
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
            bnorm=self.bnorm,
            lnorm=self.lnorm
        )

    def make_lang_denses(self, inpt_size=None):
        if inpt_size==None: inpt_size = self.h_size
        self.lang_denses = nn.ModuleList([])
        # In case we're actually making multiple language predictions
        # from a single output
        lang_size = self.lang_size
        if self.max_char_seq is not None and self.max_char_seq>1:
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
                    bnorm=self.bnorm
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
        """
        super().__init__()
        self.inpt_size = inpt_size
        self.h_size = h_size
        self.lang_size = lang_size
        self.n_loops = max_char_seq
        self.n_outlayers = n_outlayers
        self.h_mult = h_mult
        self.lnorm = lnorm
        self.drop_p = drop_p
        self.actv_fxn = actv_fxn
        self.lstm = ContainedLSTM(
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
            lnorm=self.lnorm
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
            actn = torch.ones(*x.shape[:2], self.actn_size, requires_grad=True).float()
            # Language
            lang = torch.ones(*x.shape[:2], self.lang_size, requires_grad=True).float()
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
        actn = torch.ones((x.shape[0], self.actn_size), requires_grad=True).float()
        lang = torch.ones((x.shape[0], self.lang_size), requires_grad=True).float()
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
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
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
        return self.output_fxn(actn.reshape(b,s,-1)), langs

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
        assert self.bnorm == False,\
            "bnorm must be False. it does not work with Recurrence!"

        # Convs
        cnn = VaryCNN(*args, **kwargs)
        self.shapes = cnn.shapes
        self.features = cnn.features

        # LSTM
        self.flat_size = cnn.flat_size
        self.lstm = nn.LSTMCell(self.flat_size+self.h_size, self.h_size)

        self.make_actn_dense()
        self.make_lang_denses()
        # Memory
        if self.lnorm:
            self.layernorm_c = nn.LayerNorm(self.h_size)
            self.layernorm_h = nn.LayerNorm(self.h_size)
        self.h = None
        self.c = None
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
        self.h = torch.zeros(batch_size, self.h_size).float()
        self.c = torch.zeros(batch_size, self.h_size).float()
        # Ensure memory is on appropriate device
        if self.is_cuda:
            self.h.to(self.get_device())
            self.c.to(self.get_device())
        self.prev_hs = [self.h]
        self.prev_cs = [self.c]

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
        h = self.h*mask
        c = self.c*mask
        return h,c

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
        if self.is_cuda:
            self.h.to(self.get_device())
            self.c.to(self.get_device())

    def step(self, x, cdtnl, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
                a single step of observations
            cdtnl: torch FloatTensor (B, E)
                the conditional latent vectors
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor (B, L)
        """
        if x.is_cuda:
            self.h = self.h.to(x.get_device())
            self.c = self.c.to(x.get_device())
        fx = self.features(x)
        fx = fx.reshape(len(x), -1) # (B, N)
        cat = torch.cat([fx, cdtnl], dim=-1)
        self.h, self.c = self.lstm( cat, (self.h, self.c) )
        if self.lnorm:
            self.c = self.layernorm_c(self.c)
            self.h = self.layernorm_h(self.h)
        langs = []
        for dense in self.lang_denses:
            langs.append(dense(self.h))
        return self.output_fxn(self.actn_dense(self.h)), langs

    def forward(self, x, dones, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
            dones: torch Long Tensor (B, S)
                the done signals for the environment. the h and c
                vectors are reset when encountering a done signal
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
        if x.is_cuda:
            dones = dones.to(x.get_device())
        cb = self.cdtnl_batch.repeat_interleave(len(x)//self.n_envs)
        for s in range(seq_len):
            actn, lang = self.step(x[:,s], cdtnl[cb])
            actns.append(actn.unsqueeze(1))
            if self.n_lang_denses == 1:
                lang = lang[0].unsqueeze(0).unsqueeze(2) # (1, B, 1, L)
            else:
                lang = torch.stack(lang, dim=0).unsqueeze(2)# (N, B, 1, L)
            langs.append(lang)
            self.h, self.c = self.partial_reset(dones[:,s])
            self.prev_hs.append(self.h.detach().data)
            self.prev_cs.append(self.c.detach().data)
        return (
            torch.output_fxn(torch.cat(actns, dim=1)),
            torch.cat(langs, dim=2)
        )

class DoubleVaryLSTM(VaryLSTM):
    """
    A model with two LSTMs. One for each the language and action outputs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_lstms = 2
        self.lstm0 = self.lstm
        size = self.h_size
        if self.skip_lstm: 
            size = self.flat_size+2*self.h_size
        self.lstm1 = nn.LSTMCell(size, self.h_size)
        self.reset(1)
        self.make_actn_dense()
        self.make_lang_denses()

    def reset(self, batch_size=1):
        """
        Resets the memory vectors

        Args:
            batch_size: int
                the size of the incoming batches
        Returns:
            None
        """
        self.hs = [ ]
        self.cs = [ ]
        for i in range(self.n_lstms):
            self.hs.append(torch.zeros(batch_size, self.h_size).float())
            self.cs.append(torch.zeros(batch_size, self.h_size).float())
        # Ensure memory is on appropriate device
        if self.is_cuda:
            for i in range(self.n_lstms):
                self.hs[i] = self.hs[i].to(self.get_device())
                self.cs[i] = self.cs[i].to(self.get_device())
        self.prev_hs = [self.hs]
        self.prev_cs = [self.cs]

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
        hs = [h*mask for h in self.hs]
        cs = [c*mask for c in self.cs]
        return hs,cs

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
        device = self.get_device()
        if self.is_cuda:
            self.hs = [h.detach().data.to(device) for h in self.hs]
            self.cs = [c.detach().data.to(device) for c in self.cs]
        else:
            self.hs = [h.detach().data for h in self.hs]
            self.cs = [c.detach().data for c in self.cs]

    def step(self, x, cdtnl, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
                a single step of observations
            cdtnl: torch FloatTensor (B, E)
                the conditional latent vectors
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
        cat = torch.cat([fx, cdtnl], dim=-1)

        h0, c0 = self.lstm0( cat, (self.hs[0], self.cs[0]) )
        if self.lnorm:
            c0 = self.layernorm_c(c0)
            h0 = self.layernorm_h(h0)
        inpt = h0
        if self.skip_lstm: inpt = torch.cat([cat,inpt],dim=-1)
        h1, c1 = self.lstm1( inpt, (self.hs[1], self.cs[1]) )
        if self.lstm_lang_first:
            langs = []
            for dense in self.lang_denses:
                langs.append(dense(h0))
            actn = self.actn_dense(h1)
        else:
            langs = []
            for dense in self.lang_denses:
                langs.append(dense(h1))
            actn = self.actn_dense(h0)
        self.hs = [h0, h1]
        self.cs = [c0, c1]
        return self.output_fxn(actn), langs

    def forward(self, x, dones, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
            dones: torch Long Tensor (B, S)
                the done signals for the environment. the h and c
                vectors are reset when encountering a done signal
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
        if x.is_cuda:
            dones = dones.to(x.get_device())
        cb = self.cdtnl_batch.repeat_interleave(len(x)//self.n_envs)
        for s in range(seq_len):
            actn, lang = self.step(x[:,s], cdtnl[cb])
            actns.append(actn.unsqueeze(1))
            if self.n_lang_denses == 1:
                lang = lang[0].unsqueeze(0).unsqueeze(2) # (1, B, 1, L)
            else:
                lang = torch.stack(lang, dim=0).unsqueeze(2)# (N, B, 1, L)
            langs.append(lang)
            self.hs, self.cs = self.partial_reset(dones[:,s])
            self.prev_hs.append([h.detach().data for h in self.hs])
            self.prev_cs.append([c.detach().data for c in self.cs])
        return (
            self.output_fxn(torch.cat(actns, dim=1)),
            torch.cat(langs, dim=2)
        )


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
        assert self.bnorm == False,\
            "bnorm must be False. it does not work with Recurrence!"
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
    """
    A recurrent LSTM model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.bnorm == False,\
            "bnorm must be False. it does not work with Recurrence!"

        # Convs
        cnn = VaryCNN(*args, **kwargs)
        self.shapes = cnn.shapes
        self.features = cnn.features

        # Linear Projection
        self.flat_size = cnn.flat_size
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
        self.reset(batch_size=1)
        max_seq_len = 128
        self.register_buffer(
            "fwd_mask",
            get_transformer_fwd_mask(s=max_seq_len)
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
        pass

    def step(self, x, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            actn: torch Float Tensor (B, K)
            langs: list of torch Float Tensor (B, L)
        """
        fx = self.features(x)
        fx = self.proj(fx) # (B, H)
        self.prev_hs.append(fx)
        encs = torch.stack(list(self.prev_hs), dim=1)
        encs = self.pos_enc(encs)
        slen = encs.shape[1]
        encs = self.encoder( encs, self.fwd_mask[:slen,:slen] )
        if self.lnorm:
            encs = self.layernorm(encs[:,-1])
        langs = []
        for dense in self.lang_denses:
            langs.append(dense(encs))
        return self.output_fxn(self.actn_dense(encs)), langs

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
            langs: torch FloatTensor (N,B,S,L)
        """
        seq_len = x.shape[1]
        self.prev_hs = collections.deque(maxlen=self.seq_len)
        b,s,c,h,w = x.shape
        fx = self.features(x.reshape(-1,c,h,w)).reshape(b*s,-1)
        fx = self.proj(fx).reshape(b,s,-1)
        encs = self.pos_enc(fx)
        encs = self.encoder( encs, self.fwd_mask[:s,:s] )
        if self.lnorm:
            encs = self.layernorm(encs)
        encs = encs.reshape(b*s,-1)
        actns = self.actn_dense(encs).reshape(b,s,-1)
        langs = []
        for dense in self.lang_denses:
            langs.append(dense(encs).reshape(b,s,-1))
        return self.output_fxn(actns), torch.stack(langs,dim=0)

class ConditionalLSTM(CoreModule):
    """
    This LSTM is used to process conditional sentences into a single
    latent vector.
    """
    def __init__(self, h_size, *args, **kwargs):
        """
        h_size: int
            the hidden dimension size
        """
        super().__init__()
        self.h_size = h_size
        self.embs = nn.Embedding(CDTNL_LANG_SIZE, self.h_size)
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


