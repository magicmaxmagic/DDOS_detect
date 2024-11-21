import numpy as np

from math import log
import numpy as np

from math import log
from scipy.optimize import minimize

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import sys



def grimshaw(peaks:np.array, threshold:float, num_candidates:int=10, epsilon:float=1e-8):
    ''' The Grimshaw's Trick Method

    The trick of thr Grimshaw's procedure is to reduce the two variables 
    optimization problem to a signle variable equation. 

    Args:
        peaks: peak nodes from original dataset. 
        threshold: init threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform

    Returns:
        gamma: estimate
        sigma: estimate
    '''
    min = peaks.min()
    max = peaks.max()
    mean = peaks.mean()

    if abs(-1 / max) < 2 * epsilon:
        epsilon = abs(-1 / max) / num_candidates

    a = -1 / max + epsilon
    b = 2 * (mean - min) / (mean * min)
    c = 2 * (mean - min) / (min ** 2)

    candidate_gamma = solve(function=lambda t: function(peaks, threshold), 
                            dev_function=lambda t: dev_function(peaks, threshold), 
                            bounds=(a + epsilon, -epsilon), 
                            num_candidates=num_candidates
                            )
    candidate_sigma = solve(function=lambda t: function(peaks, threshold), 
                            dev_function=lambda t: dev_function(peaks, threshold), 
                            bounds=(b, c), 
                            num_candidates=num_candidates
                            )
    candidates = np.concatenate([candidate_gamma, candidate_sigma])

    gamma_best = 0
    sigma_best = mean
    log_likelihood_best = cal_log_likelihood(peaks, gamma_best, sigma_best)

    for candidate in candidates:
        if candidate == 0: continue
        gamma = np.log(1 + candidate * peaks).mean()
        sigma = gamma / candidate
        log_likelihood = cal_log_likelihood(peaks, gamma, sigma)
        if log_likelihood > log_likelihood_best:
            gamma_best = gamma
            sigma_best = sigma
            log_likelihood_best = log_likelihood

    return gamma_best, sigma_best


def function(x, threshold):
    s = 1 + threshold * x
    u = 1 + np.log(s).mean()
    v = np.mean(1 / s)
    return u * v - 1


def dev_function(x, threshold):
    s = 1 + threshold * x
    u = 1 + np.log(s).mean()
    v = np.mean(1 / s)
    dev_u = (1 / threshold) * (1 - v)
    dev_v = (1 / threshold) * (-v + np.mean(1 / s ** 2))
    return u * dev_v + v * dev_u


def obj_function(x, function, dev_function):
    m = 0
    n = np.zeros(x.shape)
    for index, item in enumerate(x):
        y = function(item)
        m = m + y ** 2
        n[index] = 2 * y * dev_function(item)
    return m, n


def solve(function, dev_function, bounds, num_candidates):
    step = (bounds[1] - bounds[0]) / (num_candidates + 1)
    x0 = np.arange(bounds[0] + step, bounds[1], step)
    optimization = minimize(lambda x: obj_function(x, function, dev_function), 
                            x0, 
                            method='L-BFGS-B', 
                            jac=True, 
                            bounds=[bounds]*len(x0)
                            )
    x = np.round(optimization.x, decimals=5)
    return np.unique(x)


def cal_log_likelihood(peaks, gamma, sigma):
    if gamma != 0:
        tau = gamma/sigma
        log_likelihood = -peaks.size * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * peaks)).sum()
    else: 
        log_likelihood = peaks.size * (1 + log(peaks.mean()))
    return log_likelihood

# num_candidates to data.size//10?
def pot(data:np.array, risk:float=1e-4, init_level:float=0.90, num_candidates:int=10, epsilon:float=1e-8) -> float:
    ''' Peak-over-Threshold Alogrithm

    References: 
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
    Discovery and Data Mining. 2017.

    Args:
        data: data to process
        risk: detection level
        init_level: probability associated with the initial threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform
    
    Returns:
        z: threshold searching by pot
        t: init threshold 
    '''
    # Set init threshold
    t = np.sort(data)[int(init_level * data.size)]
    peaks = data[data > t] - t

    # Grimshaw
    gamma, sigma = grimshaw(peaks=peaks, 
                            threshold=t, 
                            num_candidates=num_candidates, 
                            epsilon=epsilon
                            )

    # Calculate Threshold
    r = data.size * risk / peaks.size
    if gamma != 0: #
        z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
    else: 
        z = t - sigma * log(r)

    return z, t



time = '_time'
delay = '_value'
session = '_measurement'
tos = 'agentID'

csv_headers = [time, delay, session, tos] # Type of Service: in IP header
file_path = 'data/data_tcp_24h_120s_attack.csv'
n = 51000 # First n rows to plot

df = pd.read_csv(file_path, usecols=csv_headers, low_memory=True) # Chunks
df = df.reset_index()  # Makes sure indexes pair with number of rows
# df.head() # View 5 first rows
df = df.head(n)

# Calcul du percentile
percentile = np.percentile(df['_value'], 99.8)
df = df[df['_value'] < percentile]

# Histogramme
ax = df.plot.hist(column='_value', bins=10000, range=[0, 40], alpha=0.5)

# Plot des latences dans le temps
df.plot(x='_time', y=['_value'])
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d ms'))
plt.gcf().autofmt_xdate()  # Formatage des dates en ISO 8601
plt.show()

risk = 1e-4
init_level = 0.90
num_candidates = 10
epsilon = 1e-8

# Modèle POT
z, t = pot(df['_value'], risk, init_level, num_candidates, epsilon)

# Plot avec seuil
fig, ax = plt.subplots(figsize=(20, 10))

ax.plot(df['_value'], label='Latency (ms)', alpha=0.6, color='#6f9b92', zorder=10)
ax.axhline(y=z, xmin=0.045, xmax=0.955, linewidth=3, color='#FCFDDE', label='Threshold', zorder=5)
ax.fill_between(x=range(df['_value'].size), y1=0, y2=z, facecolor='#656A4C', alpha=0.2, zorder=1)

ax.set_facecolor('#040404')
fig.set_facecolor('#040404')

ax.spines['top'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')

ax.spines['top'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)

ax.tick_params(axis='x', colors='white', labelsize=13)
ax.tick_params(axis='y', colors='white', labelsize=13)

plt.show()