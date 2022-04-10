import numpy as np

# granular propose

def phi_func(x, mu):
    return np.exp(-(x - mu) ** 2)


def t_func(phi, sigma):
    if (np.all(phi < 0)) or (np.all(phi > 1)):
        raise AttributeError('Invalid phi value. Expected 0 <= phi <= 1')
    elif sigma <= 0:
        raise AttributeError('Expected sigma > 0')
    else:
        return phi ** (1 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

# granular characteristic functions

def chi_func(g,x, mu):

  if g <= phi_func(x, mu):
    return 1
  elif g > phi_func(x, mu):
    return 0
  else:
    raise AttributeError(f"g = {g}; x = {x}; μ = {mu}; σ = {sigma}")

def q_func(a,i):

  return 1 - (a+1)**(-i)


# reverced functions

def phi_func_reversed(phi, mu):
    if (np.all(phi < 0)) or (np.all(phi > 1)):
        raise AttributeError('Invalid phi value. Expected 0 <= phi <= 1')
    else:
        return np.sqrt(np.log(phi) * (-1)) + mu

# dence function

def norm_dence(x, mu, sigma):

  return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

