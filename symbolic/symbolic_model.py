
import torch
import numpy as np

class MODEL_TYPES:
    LOG = "log" # beta[0] * log(x+beta[1]) + beta[2]
    INVERSE = "inverse" # beta[0] / (x+beta[1]) + beta[2]

class NOISE_TYPES:
    COUNT = "count" # fxn( x+noise )
    NEURAL = "neural" # fxn( x ) + noise

class SymbolicModel:
    def __init__(self, model_type=MODEL_TYPES.LOG,
                       betas=[1,1,1],
                       noise_stds={
                           NOISE_TYPES.COUNT: 1,
                           NOISE_TYPES.NEURAL: 1}):
        """
        Args:
            model_type: str ("log" or "inverse")
                the form of the model spacing.
                    "log":     -beta[0] * log(beta[1]*x+beta[2]) + beta[3]
                    "inverse": beta[0] / (beta[1]*x+beta[2]) + beta[3]
            betas: list of floats
                the parameters for the model
            noise_stds: dict
                "neural": float
                    the standard deviation of the outer gaussian noise
                    i.e. fxn( x ) + noise
                "count": float
                    the standard deviation of the inner gaussian noise
                    i.e. fxn( x + noise )
        """
        self.model_type = model_type
        self.betas = betas
        self.noise_stds = noise_stds

        if self.model_type == MODEL_TYPES.LOG:
            self.increment = self.log_fxn
            self.inverse = self.log_inverse
        elif self.model_type == MODEL_TYPES.INVERSE:
            self.increment = self.inv_fxn
            self.inverse = self.inv_inverse

    def log_fxn(self, x):
        """
        Args:
            x: ndarray float (m, n, ...)

        Returns:
            log_x: ndarray float (m, n, ...)
        """
        if NOISE_TYPES.COUNT in self.noise_stds:
            noise = np.random.normal(
                scale=self.noise_stds[NOISE_TYPES.COUNT],
                size=x.shape
            )
            x = x + noise
        beta = self.betas
        log_x = - beta[0] * np.log(beta[1]*x + beta[2]) + beta[3]
        log_x[(log_x!=log_x)|(x<=0)] = 0
        if NOISE_TYPES.NEURAL in self.noise_stds:
            noise = np.random.normal(
                scale=self.noise_stds[NOISE_TYPES.NEURAL],
                size=log_x.shape
            )
            log_x = log_x + noise
        return log_x

    def inv_fxn(self, x):
        """
        Args:
            x: ndarray float (m, n, ...)

        Returns:
            inv_x: ndarray float (m, n, ...)
        """
        if NOISE_TYPES.COUNT in self.noise_stds:
            noise = np.random.normal(
                scale=self.noise_stds[NOISE_TYPES.COUNT],
                size=x.shape
            )
            x = x + noise
        beta = self.betas
        inv_x = beta[0] / (beta[1]*x + beta[2]) + beta[3]
        inv_x[(inv_x!=inv_x)|(x<=0)] = 0
        if NOISE_TYPES.NEURAL in self.noise_stds:
            noise = np.random.normal(
                scale=self.noise_stds[NOISE_TYPES.NEURAL],
                size=inv_x.shape
            )
            inv_x = inv_x + noise
        return inv_x

    def log_inverse(self, log_x):
        """
        Inverts the log_fxn

        Args:
            log_x: ndarray float (m, n, ...)

        Returns:
            x: ndarray float (m, n, ...)
        """
        if NOISE_TYPES.NEURAL in self.noise_stds:
            noise = np.random.normal(
                scale=self.noise_stds[NOISE_TYPES.NEURAL],
                size=log_x.shape
            )
            log_x = log_x + noise
        beta = self.betas
        x = (np.exp((log_x-beta[3])/(-beta[0]))-beta[2])/beta[1]
        if NOISE_TYPES.COUNT in self.noise_stds:
            noise = np.random.normal(
                scale=self.noise_stds[NOISE_TYPES.COUNT],
                size=x.shape
            )
            x = x + noise
        return x

    def inv_inverse(self, inv_x):
        """
        Inverts the inv_fxn

        Args:
            inv_x: ndarray float (m, n, ...)

        Returns:
            x: ndarray float (m, n, ...)
        """
        if NOISE_TYPES.NEURAL in self.noise_stds:
            noise = np.random.normal(
                scale=self.noise_stds[NOISE_TYPES.NEURAL],
                size=inv_x.shape
            )
            inv_x = inv_x + noise
        beta = self.betas
        x = ( beta[0]/(inv_x-beta[3]) - beta[2] )/beta[1]
        if NOISE_TYPES.COUNT in self.noise_stds:
            noise = np.random.normal(
                scale=self.noise_stds[NOISE_TYPES.COUNT],
                size=x.shape
            )
            x = x + noise
        return x

    def __call__(self, x):
        return self.increment(x)#Encodes from count space to neural space

    def decode(self, x):
        return self.inverse(x) #Inverts from neural space to count space
    
    
    
def count_up_count_down(model, target_num, continuous=False, n_samps=1):
    """
    This function uses the model to first count up target_num steps
    and then count down till the model reaches 0

    Args:
        model: SymbolicModel
        target_num: int
            the maximum count
        continuous: bool
            if true, will use a continuous representation of the count.
            if false, will use a discrete representation of the count.
        n_samps: int
            the number of samples to run

    Returns:
        dict(
            up_count: ndarray (n_samps,)
                the model's counting value after target_num steps
            down_count: ndarray (n_samps,)
                the number of steps taken to reach zero stepping down
                from up_count
            history: dict
                "neural": dict
                    "up": list of ndarray [(n_samps,), ...]
                        the neural history during the count up phase
                    "down": list of ndarray [(n_samps,), ...]
                        the neural history during the count down phase
                "count": dict
                    "up": list of ndarray [(n_samps,), ...]
                        the count history during the count up phase
                    "down": list of ndarray [(n_samps,), ...]
                        the count history during the count down phase
    """
    count = np.zeros((n_samps,))
    neural = model(np.ones(n_samps,))
    
    up_neurals = []
    up_counts =  []
    for i in range(1, target_num+1):
        if i == 1:
            up_neurals.append(neural)
            up_counts.append(model.decode(up_neurals[-1]))
        else:
            inc = model(up_counts[-1]+1)
            up_neurals.append(up_neurals[-1]+inc)
            up_counts.append(model.decode(up_neurals[-1]-up_neurals[-2]))
            
        if not continuous:
            up_counts[-1] = np.round(up_counts[-1])
            
    up_count = up_counts[-1].copy()
    up_neural = up_neurals[-1].copy()

    # Count Down
    down_count = np.zeros_like(up_count) - 1
    down_neural = np.zeros_like(down_count)
    down_counts =  [ up_counts[-1] ]
    down_neurals = [ up_neurals[-1] ]
    loop = 1
    while np.any(down_counts[-1]>0) and loop < target_num*2+20:
        dec = model( down_counts[-1] )
        down_neurals.append( down_neurals[-1] - dec )
        
        down_counts.append( model.decode(dec)-1 )
        if not continuous:
            down_counts[-1] = np.round(down_counts[-1])
            
        idx = (down_counts[-1]<=0)&(down_count==-1)
        down_count[idx] = loop
        down_neural[idx] = down_neurals[-1][idx]
        loop += 1
        
    history = {
        "neural": { "up": up_neurals, "down": down_neurals },
        "count":  { "up": up_counts, "down": down_counts }
    }

    return {
        "up_count": up_count,
        "up_neural": up_neural,
        "down_count": down_count,
        "down_neural": down_neural,
        "history": history,
    }



