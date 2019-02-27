
# coding: utf-8

# # Using Adaline as a tapped-delay adaptive filter for demonstrating 50 Hz noise cancellation in a computer generated white noise signal

# The notations used in the code are similar to that in the example of a noise cancellation system in EEG, Neural Network design, Hagan et al., Chapter 10

#/usr/bin/python

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

parser = argparse.ArgumentParser(description='Adaline for Adaptive Noise Filtering')

parser.add_argument('-nt', '--num_taps', type=int, default=100,
                    help='Number of taps for model.')
parser.add_argument('-d', '--duration', type=int, default=500,
                    help='Number of samples in signal.')
parser.add_argument('-lr', '--learning_rate', type=float,
                    default=0.01,
                    help='Learning rate for the model.')

args = parser.parse_args()


# **Utility Functions**

# Helper functions to generate required signals.



def input_signal_generator(mean, std, num_samples):
    """ 
    Function to generate random white noise input signal. 
    
    Parameters
    -------------------
    mean: int
        Mean value of signal
    std: int
        Standard deviation of signal
    num_samples: 
        No. of samples to be drawn from the Gaussian distribution
    
    Returns
    ------------------
    signal: np.ndarray
        White Noise Signal
    """ 
    signal = np.random.normal(loc=mean, scale=std, size=num_samples)
    signal = np.reshape(signal,(signal.shape[0],1))
    
    return signal

def generate_noise_sine_wave(freq, samp_freq, duration, phase_shift):
    """ 
    Generates sinusoidal noise. 
    
    Parameters
    -------------------
    freq: int
        Frequency of Signal
    samp_freq: int
        Sampling frequency of signal
    duration: 
        No. of samples to be generated for the signal
    phase_shift: float
        Phase of the sine wave
    
    Returns
    -------------------
    sinusoid: np.ndarray
        Sinusoidal wave of specified frequency and phase shift and duration 
    """
    k = np.arange(duration) # No. of samples in the signal
    sinusoid = np.sin((2*np.pi*k*(freq/samp_freq)) + phase_shift)
    sinusoid = np.reshape(sinusoid,(sinusoid.shape[0],1))
    
    return sinusoid

def generate_filtered_noise(attenuation_factor, phase_shift, duration=1000):
    """ Generates filtered (attenuated and phase-shifted) signal from noise. 
    Parameters
    -------------------
    attenuation_factor: int
        To attenuate amplitude of signal
    phase_shift: float
        To add phase_shift to pure sinusoidal noise
    duration: int
        No. of samples to be generated for the signal
    
    Returns
    ------------------
    m: np.ndarray
        Attenuated and phase shifted (i.e. filtered) noise    
    """
    m = generate_noise_sine_wave(freq=50, samp_freq=500, duration=duration, phase_shift= phase_shift)
    m = m / attenuation_factor
    m = np.reshape(m,(m.shape[0],1))
    return m

def signal_after_tap_delay(y, k, R):
    """ Utility function for tapped_delay_line. 
    
    Parameters
    -------------
    y: np.ndarray
        Input signal
    k: int
        index from which to start 
    R: int
        number of tap delays 
    """
    start, end = k-R, k+1
    
    if start < 0:
        padded_signal = np.pad(y, (R-k, 0), 'constant')
        start, end = 0, R+1
        y = padded_signal
    
    output_signal = np.flip(y[start: end], axis=0)    
        
    return output_signal

def tapped_delay_line(input_signal, tap): 
    """ Generates input to Adaline part of the filter. 
    
    Parameters
    -------------
    input_signal: np.ndarray
        Input signal
    tap: int
        number of delay elements 
    """
    y = np.reshape(input_signal, (input_signal.shape[0],))
    z = list()
    for k in range(len(y)):
        z.append(signal_after_tap_delay(y, k, tap))

    output_signal = np.vstack(z)
    
    return output_signal

def generate_dataset_for_adaptive_filter(noise, contaminated, tap):
    """ Generates delayed input to adaptive filter.
    
    Parameters
    -------------------
    noise: np.ndarray
        input to tapped delay line
    tap: int
        No. of delay elements
    contaminated: np.ndarray
        Desired output for adaptive filter
        
    Returns
    --------------------
    X_train.values: np.ndarray
        training examples for adaptive filter adaline 
    y_train: np.ndarray
        targets for adaptive filter adaline
    """
    delayed_input = tapped_delay_line(noise, tap)
    headers = ['p' + str(i+1) for i in range(tap+1)]
    X_train = pd.DataFrame(delayed_input, columns=headers)
    y_train = contaminated
    
    return X_train.values, y_train

def generate_contaminated_signal(input_, filtered_noise):
    """ Generates contaminated signal. 
    
    Parameters
    -------------
    input_: np.ndarray
        input signal 
    filtered_noise: np.ndarray
        filtered noise to contaminate input signal
    """
    return (input_ + filtered_noise)


# **Model**

# Model representing the Adaptive Filter using Adaline.

class AdaptiveFilter:
    
    def __init__(self, num_taps, learning_rate):
        self.num_taps = num_taps
        self.input_dim = self.num_taps + 1
        self.learning_rate = learning_rate
        
        self._initialize_weights()
        
    def _activation_function(self, weighted_sum):
        "Purelin activtion of adaline"
        return weighted_sum
    
    def _initialize_weights(self):
        """ Model initializer. """
        self.weights = 1.0 * np.random.randn(1, self.input_dim) + 0.
        self.bias = 1.0 * np.random.randn(1, 1) + 0.
        
    def fit(self, X_train, y_train):
        """ 
        Method to train Adaline.
        
        Parameters
        ----------
        X_train: pd.DataFrame
            DataFrame of features.
        y_train: np.array
            Target for each samples.
        """
        self.errors = []
        self.outputs = []
        for i in range(len(X_train)):
            x, t = X_train[i], y_train[i]
            weighted_sum = np.dot(self.weights, x) + self.bias
            a = self._activation_function(weighted_sum)
            
            e = t - a
            self.weights = self.weights + 2 * self.learning_rate * e * x.T
            self.bias = self.bias + 2 * self.learning_rate * e
            
            self.errors.append(e)
            self.outputs.append(a)
     
    @property
    def outputs(self):
        return self.__outputs
    
    @outputs.setter
    def outputs(self, outputs_):
        self.__outputs = outputs_
        
    @property
    def errors(self):
        return self.__errors
    
    @errors.setter
    def errors(self, errors_):
        self.__errors = errors_


# **Testing**

# Hyperparameters
# - NUM_TAPS: Number of delay taps in the filter.
# - DURATION: Number of samples drawn for the signal.
# - LEARNING_RATE: Learning rate of the model.

# Defining HyperParameters 
NUM_TAPS = 100
DURATION = 500
LEARNING_RATE = 1e-2

# Generating required signals for the model
s = input_signal_generator(mean=0, std=0.2, num_samples=DURATION)
v = generate_noise_sine_wave(freq=50, samp_freq=200, duration=DURATION, phase_shift=0)
m = generate_filtered_noise(attenuation_factor=10, duration=DURATION, phase_shift=(np.pi)/2)
t = generate_contaminated_signal(s,m)

# Generating model and training examples and targets
adaline = AdaptiveFilter(num_taps=NUM_TAPS, learning_rate=LEARNING_RATE)
X_train, y_train = generate_dataset_for_adaptive_filter(v, t, NUM_TAPS)
print("Shapes =", X_train.shape, y_train.shape)

# Training Model
print ("Fitting data...")
adaline.fit(X_train, y_train)

# Calculating errors
e = np.vstack(adaline.errors)
e = np.reshape(e, (DURATION,))
x = np.arange(len(e))

# Plotting results
print ("Plotting...")
rcParams['figure.figsize'] = 15,5
plt.plot(x, e, color='indigo', label='Restored signal')
plt.plot(x, s, color='red', label='Input signal')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

print ("Done.")

