from scipy.stats import norm

def PI(mean,var):
    eps = 1e-7
    y_hat = np.max(mu)
    theta = (mean - y_hat)/(var + eps)
    return np.array([norm.cdf(theta[i]) for  i in range(len(theta))])

