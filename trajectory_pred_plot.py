import matplotlib
import torch
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import modules, dataio
from stats import *


model = modules.SingleBVPNet(in_features=6, out_features=1, final_layer_factor=1.,hidden_features=512, num_hidden_layers=3)
model.cuda()
model.load_state_dict(torch.load('logs/betaVary_pre30_src10_epo110_rad015/checkpoints/model_final.pth'))


num_steps = 10
u_h = torch.ones(num_steps)*-np.pi
u_h[4:6] = np.pi/2
bel0 = torch.ones(2,num_steps)*0.5
bel_neg = torch.ones(2,num_steps+1)*0.5
bel_pos = torch.ones(2,num_steps+1)*0.5
eps = 0.2

beta1 = 0.1
beta2 = 10

goal = torch.tensor ([0.,0.])
x = torch.zeros(2,num_steps+1) # initial condition
x[:,0] = torch.tensor([0.95,0.8])
vel = 0.75
dt = 0.15 # timestep
for i in range(num_steps):
  if i > 0:
    bel_neg[:,i] = (1-eps)*bel_pos[:,i] + eps*bel_neg[:,0]
  
  act = u_h[i]
  u_star = torch.atan2(-x[1,i]+goal[1], -x[0,i]+ goal[0])

  if i > 3 and i < 6:
    act = u_star + np.pi/2
  else:
    act = u_star
  act =u_star
  obs = act - u_star

  pdf1 = truncnorm_pdf(obs,-np.pi,np.pi,loc=0, scale = beta1)
  pdf2 = truncnorm_pdf(obs,-np.pi,np.pi,loc=0, scale = beta2)
  
  bel_pos[0,i+1] = (1-eps)*(pdf1*bel_neg[0,i] / (pdf1*bel_neg[0,i]+ pdf2*bel_neg[1,i])) + eps*(bel_neg[0,i]) 
  bel_pos[1,i+1] = (1-eps)*(pdf2*bel_neg[1,i] / (pdf1*bel_neg[0,i]+ pdf2*bel_neg[1,i])) + eps*bel_neg[1,i]

  x[0,i+1] = x[0,i] + vel*torch.cos(act)*dt
  x[1,i+1] = x[1,i] + vel*torch.sin(act)*dt
bel_neg[:,-1] = (1-eps)*bel_pos[:,-1] + eps*bel_neg[:,0]


# Create a figure
fig = plt.figure(figsize=(4.5*num_steps, 0.3*num_steps))
# Get the meshgrid in the (x, y) coordinate
sidelen = 200
mgrid_coords = dataio.get_mgrid(sidelen)


# Start plotting the results
for i in range(num_steps):
  time_coords = torch.ones(mgrid_coords.shape[0], 1) * 0.2
  x_coords = torch.ones(mgrid_coords.shape[0], 1) * x[0,i]
  y_coords = torch.ones(mgrid_coords.shape[0], 1) * x[1,i]

  p_coords = torch.ones(mgrid_coords.shape[0], 1) * bel_neg[0,i]
  print(bel_pos[0,i])
  coords = torch.cat((time_coords, mgrid_coords, x_coords, y_coords, p_coords), dim=1) 
  model_in = {'coords': coords.cuda()}
  model_out = model(model_in)['model_out']
  # Detatch model ouput and reshape
  model_out = model_out.detach().cpu().numpy()
  model_out = model_out.reshape((sidelen, sidelen))
  # Unnormalize the value function
  norm_to = 0.02
  mean = 0.25
  var = 0.5
  model_out = (model_out*var/norm_to) + mean 
  # Plot the zero level sets
  model_out = (model_out <= 0.001)*1.
  # Plot the actual data
  ax = fig.add_subplot(1,num_steps, i+1)
  ax.set_title('t = %0.2f' % (i*dt))
  s = ax.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
  #print(model_out.T)
  plt.scatter(0, 0)
  plt.scatter(x[0,i], x[1,i])
  fig.colorbar(s) 

fig.savefig( 'TrajPredict.png')
#print(x)
