import numpy as np
import torch
def phi(x,mu, sig):
  return 0.5 * (1 + torch.erf((x-mu)/(sig*torch.sqrt(torch.tensor(2.)))))

def phi_inv(x, mu, sig):
  return mu + sig*torch.sqrt(torch.tensor(2.))*torch.erfinv(2*x - 1)

def normal_pdf(x, mu, sig):
  return (1/(sig*np.sqrt(2*np.pi)))*torch.exp(-0.5*(torch.pow((x-mu)/sig, 2)))

def truncnorm_ppf(p, a,b, loc = 0, scale = 1):
  std_dev = scale
  phi_a = phi(a, loc, std_dev)  
  phi_b = phi(b, loc, std_dev)
  
  input = phi_a + p*(phi_b - phi_a)

  return phi_inv(input, loc,scale)

def truncnorm_pdf(x, a, b, loc = 0, scale = 1):
  std_dev = scale
  return (1/scale)*(normal_pdf(x,loc, std_dev)/(phi(b, loc, std_dev) - phi(a,loc, std_dev)))
