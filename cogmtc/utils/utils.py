import math
import torch.nn.functional as F
import numpy as np
import torch
import json
import os
import cv2

BASELINE = -1
INEQUALITY = 0
ENGLISH = 1
PIRAHA = 2
RANDOM = 3
DUPLICATES = 4
NUMERAL = 5
ACTIONS = 6

PIRAHA_WEIGHTS = {
        3:   torch.FloatTensor([.55, .45]),
        4:   torch.FloatTensor([.4, .6]),
        5:   torch.FloatTensor([.4, .6]),
        6:   torch.FloatTensor([.4, .6]),
        7:   torch.FloatTensor([.45, .55]),
        8:   torch.FloatTensor([.3, .7]),
        9:   torch.FloatTensor([.3, .7]),
        10:  torch.FloatTensor([.3, .7]),
    }
#PIRAHA_WEIGHTS = {
#        3:   torch.FloatTensor([.6, .4]),
#        4:   torch.FloatTensor([.4, .6]),
#        5:   torch.FloatTensor([.4, .6]),
#        6:   torch.FloatTensor([.4, .6]),
#        7:   torch.FloatTensor([.5, .5]),
#        8:   torch.FloatTensor([.25, .75]),
#        9:   torch.FloatTensor([.25, .75]),
#        10:  torch.FloatTensor([.25, .75]),
#    }

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def try_key(d, key, val):
    """
    d: dict
    key: str
    val: object
        the default value if the key does not exist in d
    """
    if key in d:
        return d[key]
    return val

def load_json(file_name):
    """
    Loads a json file as a python dict

    file_name: str
        the path of the json file
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name,'r') as f:
        s = f.read()
        j = json.loads(s)
    return j

def save_json(data, file_name):
    """
    saves a dict to a json file

    data: dict
    file_name: str
        the path that you would like to save to
    """
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def resize2Square(img, size):
    """
    resizes image to a square with the argued size. Preserves the aspect
    ratio. fills the empty space with zeros.

    img: ndarray (H,W, optional C)
    size: int
    """
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: 
        return cv2.resize(img, (size, size), cv2.INTER_AREA)
    if h > w: 
        dif = h
    else:
        dif = w
    interpolation = cv2.INTER_AREA if dif > size else\
                    cv2.INTER_CUBIC
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
      mask = np.zeros((dif, dif), dtype=img.dtype)
      mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
      mask = np.zeros((dif, dif, c), dtype=img.dtype)
      mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)

def rand_sample(arr, n_samples=1, rand=None):
    """
    Uniformly samples a single element from the argued array.

    Args:
        arr: indexable sequence
        rand: None or numpy random number generator
    """
    if not isinstance(arr,list): arr = list(arr)
    if len(arr) == 0:
        print("len 0:", arr)
        return None
    if rand is None: rand = np.random
    samples = []
    perm = rand.permutation(len(arr))
    for i in range(n_samples):
        samples.append(arr[perm[i]])
    if len(samples) == 1: return samples[0]
    return samples

def get_max_key(d):
    """
    Returns key corresponding to maxium value

    d: dict
        keys: object
        vals: int or float
    """
    max_v = -np.inf
    max_k = None
    for k,v in d.items():
        if v > max_v:
            max_v = v
            max_k = k
    return max_k

def update_shape(shape, depth, kernel=3, padding=0, stride=1, op="conv"):
    """
    Calculates the new shape of the tensor following a convolution or
    deconvolution. Does not operate in place on shape.

    shape: list-like (chan, height, width)
    depth: int
        the new number of channels
    kernel: int or list-like
        size of the kernel
    padding: list-like or int
    stride: list-like or int
    op: str
        'conv' or 'deconv'
    """
    heightwidth = np.asarray([*shape[-2:]])
    if type(kernel) == type(int()):
        kernel = np.asarray([kernel, kernel])
    else:
        kernel = np.asarray(kernel)
    if type(padding) == type(int()):
        padding = np.asarray([padding,padding])
    else:
        padding = np.asarray(padding)
    if type(stride) == type(int()):
        stride = np.asarray([stride,stride])
    else:
        stride = np.asarray(stride)

    if op == "conv":
        heightwidth = (heightwidth - kernel + 2*padding)/stride + 1
    elif op == "deconv" or op == "conv_transpose":
        heightwidth = (heightwidth - 1)*stride + kernel - 2*padding
    return (depth, *heightwidth)

def sample_action(pi, rand=None):
    """
    Stochastically selects an action from the pi vectors.

    Args:
        pi: torch FloatTensor (..., N) (must sum to 1 across last dim)
            this is most likely going to be a model output vector that
            has passed through a softmax
        rand: None or numpy random number generator
    """
    if rand is None: rand = np.random
    pi = pi.cpu()
    rand_nums = torch.from_numpy(rand.random(*pi.shape[:-1]))
    cumu_sum = torch.zeros(pi.shape[:-1])
    actions = -torch.ones(pi.shape[:-1])
    for i in range(pi.shape[-1]):
        cumu_sum += pi[...,i]
        actions[(cumu_sum >= rand_nums)&(actions < 0)] = i
    return actions

def sample_numpy(pi, rand=None):
    """
    Stochastically selects an index from the pi vectors.

    Args:
        pi: ndarray (N,) (must sum to 1 across last dim)
        rand: None or random number generator
    """
    if rand is None: rand = np.random
    rand_num = rand.random()
    cumu_sum = 0
    action = 0
    for i in range(len(pi)):
        cumu_sum += pi[i]
        if cumu_sum > rand_num: return i
    return len(pi)-1

def softmax(arr):
    """
    arr: ndarray (N,)
        a single dimensional array
    """
    arr = np.exp(arr-np.max(arr))
    return arr / np.sum(arr)

def zipfian(low=1, high=9, order=1, rand=None):
    """
    Draws a single integer from low (inclusive) to high (inclusive) in
    which the probability is proportional to 1/k^order.

    Args:
        low: int (inclusive)
            the lowest possible value
        high: int (inclusive)
            the highest possible value
        order: float
            the order of the exponent to weight the probability density
            for each possible value.
        rand: None or random number generator
            if None, uses np.random instead
    Returns:
        sample: int
            returns a sample drawn from the zipfian distribution.
    """
    if low == high: return low
    assert low < high and low > 0
    probs = np.arange(low, high+1).astype("float")
    probs = 1/(probs**order)
    probs = probs/probs.sum()
    samp = sample_numpy(probs, rand=rand)
    return samp + low

def get_piraha_labels(labels, n_items):
    """
    Converts the number of items that exist in the game (not
    including the targets) to a count word in the piraha language.

    Uses the following probablities for each count word conversion.
    Probabilities taken from Frank 2008.
        number 0:
            labels: 0
            probabilities: [1]
        number 1:
            labels: 1
            probabilities: [1]
        number 2:
            labels: 2
            probabilities: [1]
        number 3:
            labels: 2,3
            probabilities: [.55, .45]
        numbers 4-6:
            labels: 2,3
            probabilities: [.4, .6]
        numbers 7:
            labels: 2,3
            probabilities: [.45, .55]
        numbers 8 and above:
            labels: 2,3
            probabilities: [.3, .7]

    Args:
        labels: torch Tensor (...,N)
            the count of the items on the board (a clone of n_items
            works just fine)
        n_items: torch Tensor (...,N)
            the count of the items on the board
    Returns:
        labels: torch LongTensor
            the updated labels. operates in place 
    """
    weights = {**PIRAHA_WEIGHTS}
    labels[n_items==1] = 1
    labels[n_items==2] = 2
    # Sample the Piraha count words with the appropriate length 
    # using weights found in Frank's 2008 "Number as a cog tech"
    min_key = np.min(list(weights.keys()))
    max_key = np.max(list(weights.keys()))
    for i in range(min_key,max_key):
        idx = (n_items==i)
        l = len(labels[idx])
        if l > 0:
            labs = torch.multinomial(weights[i], l, replacement=True)
            # samples are 0 indexed, so add 2 for the proper label
            labels[idx] = labs + 2
    # Repeat previous step for numbers greater than max specified
    idx = (n_items>=max_key)
    l = len(labels[idx])
    if l > 0:
        labs = torch.multinomial(weights[max_key], l, replacement=True)
        # samples are 0 indexed, so add 2 for the proper label
        labels[idx] = labs + 2
    return labels

def get_duplicate_labels(labels, n_items, max_targ,null_label,rand=None):
    """
    Converts the number of items that exist in the game (not
    including the targets) to a count word that is interchangeable
    with another count word meaning the same thing. For example, the
    label for the value "0" can either be 0 or 1 with equal probability.
    Similarly the label "1" can be either 2 or 3. This pattern continues
    up to the max label.

    Args:
        labels: torch Tensor (...,N)
            the count of the items on the board (a clone of n_items
            works just fine)
        n_items: torch Tensor (...,N)
            the count of the items on the board
        max_targ: int
            the maximum value to label. 
        rand: np random number generator
    Returns:
        labels: torch LongTensor
            the updated labels. operates in place 
    """
    if rand is None: rand = np.random
    rand_vals = torch.from_numpy(
        rand.random(labels.shape)*2
    ).long()
    for i in range(0,max_targ*2+1,2):
        idx = n_items==(i//2)
        labels[idx] = i+rand_vals[idx]
    if null_label is None: null_label = (max_targ+1)*2+1
    labels[n_items>max_targ] = null_label
    return labels

def get_lang_labels(n_items,
                    n_targs,
                    max_targ,
                    use_count_words,
                    max_char_seq=1,
                    base=4,
                    lang_offset=0,
                    null_label=None,
                    stop_label=None):
    """
    Determines the language labels based on the type of training.
    null_labels are only applied to English and Duplicate variants

    Args:
        n_items: torch Tensor (N,)
            the count of the items on the board
        n_targs: torch Tensor (N,)
            the count of the targs on the board
        max_targ: int
            the inclusive maximum allowed target value to be included in
            language. can usually use hyps["lang_range"][-1]
        use_count_words: int
            the type of language that the model is using
        max_char_seq: int
            the longest character sequence that the model will ever
            predict. Only matters when use_count_words is 5
        base: int
            the base of the number system if using NUMERAL models
        lang_offset: int
            the number of values that the labels should be offset by
        null_label: int or None
            if none, takes the value of 1+greatest possible label for
            use_count_words type
        stop_label: int or None
            if int, replaces base as the index to denote the end of
            a NUMERAL sequence
    Returns:
        labels: torch Tensor (N,) or (N,M)
            returns tensor of shape (N,M) when use_count_words is 5
    """
    labels = n_items.clone()
    ctype = int(use_count_words)
    # positional numeral system
    if ctype == NUMERAL:
        if null_label is None: null_label = base+2
        mcs = max_char_seq
        labels = get_numeral_labels(labels, base, mcs)
        #idx = n_items.reshape(-1)>max_targ
        #if torch.any(idx):
        #    null = (torch.ones_like(labels)*-1).reshape(-1, mcs)
        #    null[:,0] = null_label # null label
        #    null[:,1] = base # stop label
        #    og_shape = labels.shape
        #    labels = labels.reshape(-1, mcs)
        #    labels[idx] = null[idx]
        #    labels = labels.reshape(og_shape)
        if stop_label is not None:
            idx = labels==base
            labels[idx] = stop_label
            labels[~idx] += lang_offset
        else:
            labels += lang_offset
        return labels

    if ctype == ENGLISH:
        if null_label is None: null_label = max_targ-1
        labels[n_items>max_targ] = null_label
    elif ctype == INEQUALITY:
        labels[n_items<n_targs] = 0
        labels[n_items==n_targs] = 1
        labels[n_items>n_targs] = 2
    # Piraha labels
    elif ctype == PIRAHA:
        labels = get_piraha_labels(labels, n_items)
    # Random labels
    elif ctype == RANDOM:
        labels = torch.randint(0, max_targ+1, labels.shape)
        if n_items.is_cuda: labels = labels.to(DEVICE)
    # Duplicate labels
    elif ctype == DUPLICATES:
        if null_label is None: null_label = max_targ-1
        labels = get_duplicate_labels(labels,n_items,max_targ,null_label)
    elif ctype == ACTIONS:
        labels = torch.zeros_like(labels)
    labels += lang_offset
    return labels

def get_numeral_labels(n_items,numeral_base=4,char_seq_len=4):
    """
    Vectorized number base changing system. Creates a tensor of
    `char_seq_len` in the last dimension. Fills each row of the tensor
    with the digits of the new number base. Appends a stop token, which
    is represented by the `numeral_base` value, to the end of each
    sequence of digits. Fills out the rest of the spaces with -1's.

    Args:
        n_items: torch Tensor (B,N) or (N,)
            the count of the items on the board in decimal numbers
        numeral_base: int
            the base of the numeral system. 10 is decimal
        char_seq_len: int
            the longest character sequence that the model will ever
            predict.
    Returns:
        labels: torch Long Tensor (N, char_seq_len)
            sequence of digits representing the new number in base
            `numeral_base`
    """
    og_shape = n_items.shape
    if len(og_shape)>1:
        n_items = n_items.reshape(-1)
    N = n_items.shape[0]
    # negative ones denote null token
    labels = -torch.ones(N, char_seq_len)
    remains = n_items
    max_poss = numeral_base**char_seq_len - 1 # use last slot for STOP
    remains[remains>max_poss] = max_poss
    logs = torch.log(remains)/math.log(numeral_base)
    logs[remains==0] = 0
    for i in range(char_seq_len):
        floor_logs = torch.floor(logs)
        divs = numeral_base**floor_logs
        floors = torch.floor(remains/divs)
        labels[logs>=0,i] = floors[logs>=0]
        remains = remains-(floors*divs)
        logs -= 1
    argmaxes = torch.argmax((labels==-1).long(), dim=-1)
    # stop token is numeral_base
    labels[torch.arange(len(labels)).long(),argmaxes] = numeral_base
    return labels.reshape(*og_shape, char_seq_len).long()

def change_base(n, base):
    """
    Slow, non-vectorized base change code for testing.

    Args:
        n: int
            number to change base
        base: int
    """
    n = int(n)
    if n == 0: return 0
    base = int(base)
    rem = n
    chars = ""
    while rem > 0:
        new_rem = rem//base
        chars = str(rem-new_rem*base) + chars
        rem = new_rem
    return int(chars)

def basen_2base10(arr, base):
    """
    Vectorized function to change all values of base base in arr to
    base 10.

    Args:
        arr: ndarray or torch tensor
            array of base based values
        base: int
            the current base of the values in arr
    Returns:
        base10: same type as arr
            the converted values
    """
    if isinstance(arr, int): arr = np.asarray([arr])
    if type(arr) == type(np.array([])):
        mod = np.mod
        div = np.zeros(1)+10
        base10 = np.zeros_like(arr)
        floor = np.floor
    else:
        mod = torch.remainder
        div = torch.zeros(1)+10
        base10 = torch.zeros_like(arr)
        floor = torch.floor

    i = 0
    while (arr>0).sum() > 1:
        mult = base**i
        i+=1
        rem = mod(arr, div)
        add = (rem*mult).astype("int")
        base10[arr>0] += add[arr>0]
        arr = floor(arr/10)
    return base10

def pre_step_up(arr):
    """
    A function to determine what indices are less than their
    successive index. Returns a binary array.

    Args:
        arr: torch Tensor (..., N)
    Returns:
        pre_idxs: torch LongTensor (..., N)
    """
    diffs = arr[...,1:]-arr[...,:-1]
    diffs[diffs>0] = 1
    diffs[diffs!=1] = 0
    pre_idxs = torch.zeros_like(arr)
    pre_idxs[...,:-1] = diffs
    return pre_idxs

def post_step_up(arr):
    """
    A function to determine what indices are greater than their
    preceding index. Returns a binary array.

    Args:
        arr: torch Tensor (..., N)
    Returns:
        post_idxs: torch LongTensor (..., N)
    """
    diffs = arr[...,1:]-arr[...,:-1]
    diffs[diffs>0] = 1
    diffs[diffs!=1] = 0
    post_idxs = torch.zeros_like(arr)
    post_idxs[...,1:] = diffs
    return post_idxs

def describe_then_prescribe(arr, no_shifts):
    """
    Sets up a copy of arr that shifts all values within arr
    back one index at indices in which no_shifts is 0.

    Args:
        arr: torch LongTensor (..., N)
        no_shifts: torch LongTensor (..., N)
            indices that should not be shifted back a step
    Returns:
        prescribe: torch LongTensor
            a copy of arr that has shifted some of the values
    """
    prescribe = torch.empty_like(arr)
    prescribe[...,:-1] = arr[...,1:]
    prescribe[...,-1:] = arr[...,-1:]
    idx = (no_shifts==1)
    prescribe[idx] = arr[idx]
    return prescribe

def convert_numeral_array_to_numbers(numerals, STOP):
    """
    converts a numeral array (representing a single number in any base)
    to a single number. i.e. the numeral array in base 4
    [1, 3, 1, STOP, -1] will be converted to [131].

    Args:
        numerals: torch long tensor (..., B)
            the numeral array created from `get_numeral_labels`. The
            final non-negative character is ignored, all trailing
            negative charaters are ignored.
        STOP: int
            value of the stop token
    Returns:
        nums: torch float tensor (..., )
            the converted numbers
    """
    numerals = numerals.clone()
    numerals[numerals<0] = 0
    nums = torch.zeros(numerals.shape[:-1])
    tens = torch.ones_like(nums)*10
    exps = torch.argmax((numerals==STOP).long(),dim=-1)-1
    numerals[numerals==STOP] = 0
    for i in range(numerals.shape[-1]):
        nums += (tens**exps)*numerals[...,i]
        exps -= 1
    return torch.round(nums)

def get_loss_and_accs(phase,
                      actn_preds,
                      lang_preds,
                      actn_targs,
                      lang_targs,
                      drops,
                      n_targs,
                      n_items,
                      use_count_words,
                      masks=None,
                      prepender="",
                      loss_fxn=F.cross_entropy,
                      lang_p=0.5,
                      lang_size=None,
                      null_alpha=0.1):
    """
    Calculates the loss and accuracies depending on the phase of
    the training.

        Phase 0: language loss when agent drops an item
        Phase 1: action loss at all steps in rollout
        Phase 2: lang and action loss at all steps in rollout

    Args:
        phase: int - 0,1,or 2
            the phase of the training
        actn_preds: torch FloatTensor (B,S,A)
            action predictions
        lang_preds: sequence of torch FloatTensors [(B,S,L),(B,S,L),...]
            a list of language predictions. Okay to argue empty list
            or None if phase 1.
        actn_targs: torch LongTensor (B,S) or FloatTensor (B,S,...)
            action targets. The type of tensor depends on if the game
            is in a continuous or discrete action space.
        lang_targs: torch LongTensor (B,S)
            language labels
        drops: torch LongTensor (B,S)
            Ones denote steps in which the agent dropped an item, 0s
            denote all other steps
        masks: torch LongTensor or BoolTensor (B,S)
            Used to remove padding from the loss calculations.
            Ones denote locations that should be ignored. 0s denote
            locations that should be included. Be careful, this is
            the opposite of the drops.
        n_targs: torch LongTensor (B,S)
            the number of target objects on the grid at this step
            of the episode
        n_items: torch Tensor (B,S)
            the count of the items on the board
        loss_fxn: torch Module
            the loss function to calculate the loss. i.e.
            torch.nn.CrossEntropyLoss()
        use_count_words: int
            the type of language training
        prepender: str
            a string to prepend to all keys in the accs dict
        lang_p: float
            the language portion of the loss. only a factor for phase
            2 trainings
        lang_size: int or None
            only matters if using numeral system, this is the number
            of potential language classes
        null_alpha: float
            a hyperparameter to adjust how much weight should be placed
            on producing zeros for the base numeral system outputs
            following the STOP token. loss is calculated as
            loss += null_alpha*null_loss
            (It is not a proportionality parameter)
    Returns:
        loss: torch float tensor (1,)
            the appropriate loss for the phase
        accs: dict
            keys: str
            vals: float
                the appropriate label accuracies depending on the
                phase
    """
    if masks is None: masks = torch.zeros_like(n_targs)
    actn_preds = actn_preds.reshape(-1, actn_preds.shape[-1])
    if loss_fxn == F.mse_loss:
        actn_targs = actn_targs.reshape(-1, actn_targs.shape[-1])
    else:
        actn_targs = actn_targs.reshape(-1)
    drops = drops.reshape(-1)
    masks = masks.reshape(-1)
    n_targs = n_targs.reshape(-1)
    n_items = n_items.reshape(-1)
    # Phase 0: language labels when agent drops an item
    # Phase 1: action labels at all steps in rollout
    # Phase 2: combine phases 0 and 1
    loss = 0
    lang_accs = {}
    losses = {}
    if phase == 0 or phase == 2:
        loss, losses, lang_accs = calc_lang_loss_and_accs(
            lang_preds,
            lang_targs,
            drops, # determines what timesteps to train language
            masks=masks,
            categories=n_items,
            prepender=prepender,
            lang_size=lang_size,
            use_count_words=use_count_words,
            null_alpha=null_alpha
        )
    actn_accs = {}
    if phase == 1 or phase == 2:
        actn_loss, actn_accs = calc_actn_loss_and_accs(
            actn_preds,
            actn_targs,
            n_targs=n_targs,
            masks=masks,
            loss_fxn=loss_fxn,
            prepender=prepender
        )
        losses[prepender+"_actn_loss"] = actn_loss.item()
        p = lang_p if phase == 2 else 0
        loss = p*loss + (1-p)*actn_loss
    return loss, losses, {**actn_accs, **lang_accs}

def calc_actn_loss_and_accs(logits,
                            targs,
                            n_targs,
                            masks,
                            loss_fxn,
                            prepender):
    """
    Args:
        logits: torch FloatTensor (B*S,A)
            action predictions
        targs: torch LongTensor (B*S,1 or A)
            action labels
        n_targs: torch LongTensor (B*S,) or torch FloatTensor (B*S,A)
            the number of target objects on the grid at this step
            of the episode
        masks: torch LongTensor (B,S)
            Used to remove indices from the loss calculations.
            Ones denote locations that should be ignored. 0s denote
            locations that should be included.
        loss_fxn: torch Module
            the loss function to calculate the loss. i.e.
            torch.nn.CrossEntropyLoss()
        prepender: str
            a string to prepend to all keys in the accs dict
    Returns:
        loss: torch float tensor (1,)
        accs: dict
            keys: str
                accuracy types
            vals: float
                accuracies
    """
    idxs = (masks==0)
    targs = targs[idxs].to(DEVICE)
    logits = logits[idxs]
    loss = loss_fxn(logits.squeeze(), targs.squeeze())
    actn_accs = {}
    if loss_fxn == F.cross_entropy:
        with torch.no_grad():
            actn_accs = calc_accs( # accs is a dict of floats
                logits=logits,
                targs=targs,
                categories=n_targs[idxs],
                prepender=prepender+"_actn"
            )
    return loss, actn_accs

def calc_lang_loss_and_accs(preds,
                            labels,
                            drops,
                            masks,
                            categories,
                            lang_size=None,
                            use_count_words=None,
                            prepender="",
                            null_alpha=0.1):
    """
    Args:
        preds: sequence of torch FloatTensors [(B,S,L),(B,S,L),...]
            a list of language predictions. in the case that you are
            using a numeral system, L should be divisible by the
            numeral base (or C from the labels shape below)
        labels: torch LongTensor (B*S,) or (B*S*C,)
            language labels. in the case of using a numeral system, the
            shape of the labels should be equivalent to the batch by
            sequence length by number of possible numerals. -1 denotes
            outputs that should be ignored
        drops: torch LongTensor (B*S,)
            1s denote steps in which the agent dropped an item, 0s
            denote all other steps
        masks: torch LongTensor (B*S)
            Used to remove indices from the loss calculations.
            Ones denote locations that should be ignored. 0s denote
            locations that should be included.
        categories: torch long tensor (B*S,) or None
            if None, this value is ignored. Otherwise it specifies
            categories for accuracy calculations.
        lang_size: int or None
            only applies if using numeral system, this is the number
            of potential prediction classes. argue None if using
            one-hot system
        prepender: str
            a string to prepend to all keys in the accs dict
        null_alpha: float
            a hyperparameter to adjust how much weight should be placed
            on producing zeros for the base numeral system outputs
            following the STOP token.
            loss is calculated as loss += null_alpha*null_loss
            (It is not a proportionality parameter)
    Returns:
        loss: torch float tensor (1,)
        losses: dict
            keys: str
                loss types
            vals: float
                losses
        accs: dict
            keys: str
                accuracy types
            vals: float
                accuracies
    """
    accs_array = []
    losses_array = []
    labels = labels.reshape(-1)
    # Used for numeral labels
    if drops.shape[0]!=labels.shape[0]:
        n = labels.shape[0]//drops.shape[0]
        drops = drops.repeat_interleave(n)
        categories = categories.repeat_interleave(n)
        masks = masks.repeat_interleave(n)
    dmasks = (drops.float()>=1)&(masks==0)
    idxs = dmasks&(labels>=0)
    x = preds.mean(0).argmax(-1).reshape(-1)
    null_idxs = None
    if use_count_words==5: null_idxs = dmasks&(labels<0)
    categories = categories[idxs]
    labels = labels[idxs].to(DEVICE)
    loss = 0
    for j,lang in enumerate(preds):
        if null_idxs is not None:
            lang = lang.reshape(-1, lang_size)
            if null_alpha > 0:
                nulls = lang[null_idxs]
                null_loss = F.mse_loss(nulls, torch.zeros_like(nulls))
                loss += null_alpha*null_loss/lang.shape[-1]
        else:
            lang = lang.reshape(-1, lang.shape[-1])
        lang = lang[idxs]
        loss += F.cross_entropy(lang, labels)
        with torch.no_grad():
            accs = calc_accs( # accs is a dict of floats
                logits=lang,
                targs=labels,
                categories=categories,
                prepender=prepender+"_lang"
            )
            accs_array.append(accs)

            losses = calc_losses( # accs is a dict of floats
                logits=lang,
                targs=labels,
                categories=categories,
                prepender=prepender+"_lang"
            )
            losses_array.append(losses)
    losses = avg_over_dicts(losses_array)
    accs = avg_over_dicts(accs_array)
    return loss, losses, accs

def avg_over_dicts(dicts_array):
    """
    This is a helper function to average over the keys in an array
    of dicts. The result is a dict with the same keys as every
    dict in the argued array, but the values are averaged over each
    dict within the argued array.

    Args:
        dicts_array: list of dicts
            this is a list of dicts. Each dict must consist of str
            keys and float or int vals. Each dict must have the
            same set of keys.
    Returns:
        avgs: dict
            keys: str
                same keys as all dicts in dicts_array
            vals: float
                the average over all dicts in the accs array for
                the corresponding key
    """
    if len(dicts_array) == 0: return dict()
    avgs = {k: 0 for k in dicts_array[0].keys()}
    for k in avgs.keys():
        avg = 0
        for i in range(len(dicts_array)):
            avg += dicts_array[i][k]
        avgs[k] = avg/len(dicts_array)
    return avgs

def calc_accs(logits, targs, categories=None, prepender=""):
    """
    Calculates the average accuracy over the batch for each possible
    category

    Args:
        logits: torch float tensor (B, N, K)
            the model predictions. the last dimension must be the
            same number of dimensions as possible target values.
        targs: torch long tensor (B, N)
            the targets for the predictions
        categories: torch long tensor (B, N) or None
            if None, this value is ignored. Otherwise it specifies
            categories for accuracy calculations.
        prepender: str
            a string to prepend to all keys in the accs dict
    Returns:
        accs: dict
            keys: str
                <prepender>_acc: float
                    the average accuracy over all categories
                <prepender>_acc_<category>: float
                    the average accuracy over this particular
                    category. for example, if one of the categories
                    is named 1, the key will be "1" and the value
                    will be the average accuracy over that
                    particular category.
    """
    if prepender!="" and prepender[-1]!="_": prepender = prepender+"_"
    prepender = prepender + "acc"
    logits = logits.reshape(-1, logits.shape[-1])
    try:
        argmaxes = torch.argmax(logits, dim=-1).reshape(-1)
    except:
        print("logits:", logits)
        return { prepender: 0 }
    targs = targs.reshape(-1)
    acc = (argmaxes.long()==targs.long()).float()
    accs = {
        prepender: acc.mean().item()
    }
    if len(argmaxes) == 0: return accs
    pre = prepender + "lbl_"
    targ_types = {*targs.cpu().data.numpy()}
    for t in targ_types:
        idxs = targs==t
        if idxs.float().sum() == 0: continue
        accs[pre+str(t)] = acc[idxs].mean().item()
    if type(categories) == torch.Tensor: # (B, N)
        categories = categories.reshape(-1).data.long()
        pre = prepender + "ctg_"
        cats = {*categories.numpy()}
        for cat in cats:
            idxs = categories==cat
            if idxs.float().sum() <= 0: continue
            accs[pre+str(cat)] = acc[idxs].mean().item()
    return accs

def calc_losses(logits,
                targs,
                categories=None,
                prepender="",
                loss_fxn=F.cross_entropy):
    """
    Calculates the average accuracy over the batch for each possible
    category

    Args:
        logits: torch float tensor (B, N, K)
            the model predictions. the last dimension must be the
            same number of dimensions as possible target values.
        targs: torch long tensor (B, N)
            the targets for the predictions
        categories: torch long tensor (B, N) or None
            if None, this value is ignored. Otherwise it specifies
            categories for accuracy calculations.
        prepender: str
            a string to prepend to all keys in the accs dict
    Returns:
        accs: dict
            keys: str
                <prepender>_acc: float
                    the average accuracy over all categories
                <prepender>_acc_<category>: float
                    the average accuracy over this particular
                    category. for example, if one of the categories
                    is named 1, the key will be "1" and the value
                    will be the average accuracy over that
                    particular category.
    """
    if prepender!="" and prepender[-1]!="_": prepender = prepender+"_" 
    prepender = prepender + "loss"
    logits = logits.reshape(-1, logits.shape[-1])
    targs = targs.reshape(-1)
    loss = loss_fxn(logits, targs, reduction="none")
    losses = {
        prepender: loss.mean().item()
    }
    pre = prepender + "lbl_"
    targ_types = {*targs.cpu().data.numpy()}
    for t in targ_types:
        idxs = targs==t
        if idxs.float().sum() == 0: continue
        losses[pre+str(t)] = loss[idxs].mean().item()
    if type(categories) == torch.Tensor: # (B, N)
        categories = categories.reshape(-1).cpu().data.long()
        pre = prepender + "ctg_"
        cats = {*categories.numpy()}
        for cat in cats:
            idxs = categories==cat
            if idxs.float().sum() <= 0: continue
            losses[pre+str(cat)] = loss[idxs].mean().item()
    return losses

def get_transformer_fwd_mask(s):
    """
    Generates a mask that looks like this:
        0, -inf, -inf
        0,   0,  -inf
        0,   0,  0

    Args:
        s: int
            the size of each sidelength of the mask
    """
    return torch.triu(torch.ones(s,s)*float("-inf"), diagonal=1)

def max_one_hot(tensor, dim=-1):
    """
    Creates one-hot encodings of the maximum values from the argued
    tensor.

    Args:
        tensor: torch Tensor (..., N)
    Returns:
        one_hots: torch FloatTensor (..., N)
    """
    args = torch.argmax(tensor, dim=dim)[...,None]
    mask = torch.zeros_like(tensor).scatter_(-1, args, torch.ones_like(tensor))
    #mask = (tensor==torch.gather(tensor, -1, args)).float()
    return mask

def get_hook(layer_dict, key, to_numpy=True, to_cpu=False):
    """
    Returns a hook function that can be used to collect gradients
    or activations in the backward or forward pass respectively of
    a torch Module.

    Args:
        layer_dict: dict
            Can be empty

            keys: str
                names of model layers of interest
            vals: NA
        key: str
            name of layer of interest
        to_numpy: bool
            if true, the gradients/activations are returned as ndarrays.
            otherwise they are returned as torch tensors
    Returns:
        hook: function
            a function that works as a torch hook
    """
    if to_numpy:
        def hook(module, inp, out):
            layer_dict[key] = out.detach().cpu().numpy()
    elif to_cpu:
        def hook(module, inp, out):
            layer_dict[key] = out.cpu()
    else:
        def hook(module, inp, out):
            layer_dict[key] = out
    return hook

def inspect(model, X, insp_keys=set(), batch_size=500, to_numpy=True,
                                                      to_cpu=True,
                                                      no_grad=False,
                                                      verbose=False):
    """
    Get the response from the argued layers in the model as np arrays.
    If model is on cpu, operations are performed on cpu. Put model on
    gpu if you desire operations to be performed on gpu.

    Args:
        model - torch Module or torch gpu Module
        X - ndarray or FloatTensor (T,C,H,W)
        insp_keys - set of str
            name of layers activations to collect. if empty set, only
            the final output is returned.
        to_numpy - bool
            if true, activations will all be ndarrays. Otherwise torch
            tensors
        to_cpu - bool
            if true, torch tensors will be on the cpu.
            only effective if to_numpy is false.
        no_grad: bool
            if true, gradients will not be calculated. if false, has
            no impact on function.

    returns: 
        layer_outs: dict of np arrays or torch cpu tensors
            "outputs": default key for output layer
    """
    layer_outs = dict()
    handles = []
    insp_keys_copy = set()
    for key, mod in model.named_modules():
        if key in insp_keys:
            insp_keys_copy.add(key)
            hook = get_hook(layer_outs, key, to_numpy=to_numpy,
                                                 to_cpu=to_cpu)
            handle = mod.register_forward_hook(hook)
            handles.append(handle)
    if len(set(insp_keys)-insp_keys_copy) > 0:
        print("Insp keys:", insp_keys-insp_keys_copy, "not found")
    insp_keys = insp_keys_copy
    if not isinstance(X,torch.Tensor):
        X = torch.FloatTensor(X)

    # prev_grad_state is used to ensure we do not mess with an outer
    # "with torch.no_grad():" statement
    prev_grad_state = torch.is_grad_enabled() 
    if to_numpy or no_grad:
        # Turns off all gradient calculations. When returning numpy
        # arrays, the computation graph is inaccessible, as such we
        # do not need to calculate it.
        torch.set_grad_enabled(False)

    try:
        if batch_size is None or batch_size > len(X):
            if next(model.parameters()).is_cuda:
                X = X.to(DEVICE)
            preds = model(X)
            if to_numpy:
                layer_outs['outputs'] = preds.detach().cpu().numpy()
            else:
                layer_outs['outputs'] = preds.cpu()
        else:
            use_cuda = next(model.parameters()).is_cuda
            batched_outs = {key:[] for key in insp_keys}
            outputs = []
            rnge = range(0,len(X), batch_size)
            if verbose:
                rnge = tqdm(rnge)
            for batch in rnge:
                x = X[batch:batch+batch_size]
                if use_cuda:
                    x = x.to(DEVICE)
                preds = model(x).cpu()
                if to_numpy:
                    preds = preds.detach().numpy()
                outputs.append(preds)
                for k in layer_outs.keys():
                    batched_outs[k].append(layer_outs[k])
                    layer_outs[k] = None
            batched_outs['outputs'] = outputs
            if to_numpy:
                layer_outs = {k:np.concatenate(v,axis=0) for k,v in\
                                               batched_outs.items()}
            else:
                layer_outs = {k:torch.cat(v,dim=0) for k,v in\
                                         batched_outs.items()}
    except RuntimeError as e:
        print("Runtime error. Check your batch size and try using",
                "inspect with torch.no_grad() enabled")
        raise RuntimeError(str(e))

        
    # If we turned off the grad state, this will turn it back on.
    # Otherwise leaves it the same.
    torch.set_grad_enabled(prev_grad_state) 
    
    # This for loop ensures we do not create a memory leak when
    # using hooks
    for i in range(len(handles)):
        handles[i].remove()
    del handles

    return layer_outs

def get_stim_grad(model, X, layer, cell_idx, batch_size=500,
                                           layer_shape=None,
                                           to_numpy=True,
                                           ret_resps=False,
                                           verbose=True):
    """
    Gets the gradient of the model output at the specified layer and
    cell idx with respect to the inputs (X). Returns a gradient array
    with the same shape as X.

    Args:
        model: nn.Module
        X: torch FloatTensor
        layer: str
        cell_idx: int or tuple (chan, row, col)
            idx of cell (channel) of interest
        batch_size: int
            size of batching for calculations
        layer_shape: tuple of ints (chan, row, col)
            changes the shape of the argued layer to this shape if tuple
        to_numpy: bool
            returns the gradient vector as a numpy array if true
        ret_resps: bool
            if true, also returns the model responses
    Returns:
        grad: torch tensor or ndarray (same shape as X)
            the gradient of the output cell with respect to X
    """
    if verbose:
        print("layer:", layer)
    requires_grad(model, False)
    cud = next(model.parameters()).is_cuda
    device = torch.device('cuda:0') if cud else torch.device('cpu')
    prev_grad_state = torch.is_grad_enabled() 
    torch.set_grad_enabled(True)

    if model.recurrent:
        batch_size = 1
        hs = [torch.zeros(batch_size, *h_shape).to(device) for\
                                     h_shape in model.h_shapes]

    if layer == 'output' or layer=='outputs':
        layer = "sequential."+str(len(model.sequential)-1)
    hook_outs = dict()
    module = None
    for name, modu in model.named_modules():
        if name == layer:
            if verbose:
                print("hook attached to " + name)
            module = modu
            hook = get_hook(hook_outs,key=layer,to_numpy=False)
            hook_handle = module.register_forward_hook(hook)

    # Get gradient with respect to activations
    if type(X) == type(np.array([])):
        X = torch.FloatTensor(X)
    X.requires_grad = True
    resps = []
    n_loops = X.shape[0]//batch_size
    rng = range(n_loops)
    if verbose:
        rng = tqdm(rng)
    for i in rng:
        idx = i*batch_size
        x = X[idx:idx+batch_size].to(device)
        if model.recurrent:
            resp, hs = model(x, hs)
            hs = [h.data for h in hs]
        else:
            resp = model(x)
        if layer_shape is not None:
            n_samps = len(hook_outs[layer])
            hook_outs[layer] = hook_outs[layer].reshape(n_samps,
                                                        *layer_shape)
        # Outs are the activations at the argued layer and cell idx
        # for the batch
        if type(cell_idx) == type(int()):
            fx = hook_outs[layer][:,cell_idx]
        elif len(cell_idx) == 1:
            fx = hook_outs[layer][:,cell_idx[0]]
        else:
            fx = hook_outs[layer][:, cell_idx[0], cell_idx[1],
                                                  cell_idx[2]]
        fx = fx.sum()
        fx.backward()
        resps.append(resp.data.cpu())
    hook_handle.remove()
    requires_grad(model, True)
    torch.set_grad_enabled(prev_grad_state) 
    grad = X.grad.data.cpu()
    resps = torch.cat(resps,dim=0)
    if to_numpy:
        grad = grad.numpy()
        resps = resps.numpy()
    if ret_resps:
        return grad, resps
    return grad

def integrated_gradient(model, X, layer='sequential.2', chans=None,
                                                    spat_idx=None,
                                                    alpha_steps=10,
                                                    batch_size=500,
                                                    y=None,
                                                    lossfxn=None,
                                                    to_numpy=False,
                                                    verbose=False):
    """
    Returns the integrated gradient for a particular stimulus at the
    argued layer. This function always operates with the model in
    eval mode due to the need for a deterministic model. If the model
    is argued in train mode, it is set to eval mode for this function
    and returned to train mode at the end of the function. As such,
    this note is largely irrelavant, but will hopefully satisfy the
    curious or anxious ;)

    Inputs:
        model: PyTorch Deep Retina models
        X: Input stimuli ndarray or torch FloatTensor (T,D,H,W)
        layer: str layer name
        chans: int or list of ints or None
            the channels of interest. if None, uses all channels
        spat_idx: tuple of ints (row, col)
            the row and column of interest. if None, the spatial
            location of the recordings is used for each channel.
        alpha_steps: int, integration steps
        batch_size: step size when performing computations on GPU
        y: torch FloatTensor or ndarray (T,N)
            if None, ignored
        lossfxn: some differentiable function
            if None, ignored
    Outputs:
        intg_grad: ndarray or FloatTensor (T, C, H1, W1)
            integrated gradient
        gc_activs: ndarray or FloatTensor (T,N)
            activation of the final layer of the model
    """
    # Handle Gradient Settings
    # Model gradient unnecessary for integrated gradient
    requires_grad(model, False)

    # Save current grad calculation state
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(True) # Enable grad calculations
    prev_train_state = model.training
    model.eval()

    layer_idx = get_layer_idx(model, layer=layer)
    shape = model.get_shape(X.shape[-2:], layer)
    intg_grad = torch.zeros(len(X), model.chans[layer_idx],*shape)
    gc_activs = None
    model.to(DEVICE)

    if chans is None:
        chans = list(range(model.n_units))
    elif isinstance(chans,int):
        chans = [chans]

    # Handle convolutional Ganglion Cell output by replacing GrabUnits
    # coordinates for desired cell
    prev_coords = None
    if spat_idx is not None:
        if isinstance(spat_idx, int): spat_idx = (spat_idx, spat_idx)
        row, col = spat_idx
        mod_idx = get_module_idx(model, GrabUnits)
        assert mod_idx >= 0, "not yet compatible with one-hot models"
        grabber = model.sequential[mod_idx]
        prev_coords = grabber.coords.clone()
        for chan in chans:
            grabber.coords[chan,0] = row
            grabber.coords[chan,1] = col
    if batch_size is None:
        batch_size = len(X)
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    X.requires_grad = True
    idxs = torch.arange(len(X)).long()
    for batch in range(0, len(X), batch_size):
        prev_response = None
        linspace = torch.linspace(0,1,alpha_steps)
        if verbose:
            print("Calculating for batch {}/{}".format(batch, len(X)))
            linspace = tqdm(linspace)
        idx = idxs[batch:batch+batch_size]
        for alpha in linspace:
            x = alpha*X[idx]
            # Response is dict of activations. response[layer] has
            # shape intg_grad.shape
            response = inspect(model, x, insp_keys=[layer],
                                           batch_size=None,
                                           to_numpy=False,
                                           to_cpu=False,
                                           no_grad=False,
                                           verbose=False)
            if prev_response is not None:
                ins = response[layer]
                outs = response['outputs'][:,chans]
                if lossfxn is not None and y is not None:
                    truth = y[idx][:,chans]
                    outs = lossfxn(outs,truth)
                grad = torch.autograd.grad(outs.sum(), ins)[0]
                grad = grad.data.detach().cpu()
                grad = grad.reshape(len(grad), *intg_grad.shape[1:])
                l = layer
                act = (response[l].data.cpu()-prev_response[l])
                act = act.reshape(grad.shape)
                intg_grad[idx] += grad*act
                if alpha == 1:
                    if gc_activs is None:
                        gc_activs = torch.zeros(len(X),len(chans))
                    outs = response['outputs'][:,chans]
                    gc_activs[idx] = outs.data.cpu()
            prev_response={k:v.data.cpu() for k,v in response.items()}
    del response
    del grad
    if len(gc_activs.shape) == 1:
        gc_activs = gc_activs.unsqueeze(1) # Create new axis

    if prev_coords is not None:
        grabber.coords = prev_coords
    # Return to previous gradient calculation state
    requires_grad(model, True)
    # return to previous grad calculation state and training state
    torch.set_grad_enabled(prev_grad_state)
    if prev_train_state: model.train()
    if to_numpy:
        ndgrad = intg_grad.data.cpu().numpy()
        ndactivs = gc_activs.data.cpu().numpy()
        return ndgrad, ndactivs
    return intg_grad, gc_activs

