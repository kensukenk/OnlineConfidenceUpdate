import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import math
import modules, dataio
import torch
from stats import *
from anglehelper import *
import numpy as np
import imageio



if not os.path.exists('PE_game'):
    os.mkdir('PE_game')

model = modules.SingleBVPNet(in_features=6, out_features=1, final_layer_factor=1.,hidden_features=512, num_hidden_layers=3)
model.cuda()
model.load_state_dict(torch.load('../logs/HR_PE_pre40_src15_epo150_rad02_betas01_2/checkpoints/model_final.pth'))

# Create a figure
#fig = plt.figure(figsize=(5*num_times, 5*num_pos))
# Get the meshgrid in the (x, y) coordinate
sidelen = 200
mgrid_coords = dataio.get_mgrid(sidelen)

xro = [0., 0., -np.pi/2]
xhu = [0.5, 0.5]

dt = 0.01
t = 0
T = 1
delta = 0.1
vel = 0.75
omega_max = 1.1
goal = torch.tensor([0. , 0.])
beta1 = 0.1
beta2 = 2
p = 0.75
xro_t = [xro[0]]
yro_t = [xro[1]]
xhu_t = [xhu[0]]
yhu_t = [xhu[1]]
def value(xr,yr,zr,xh,yh):
  time_coords = torch.ones(mgrid_coords.shape[0], 1) * 1. 
  xr = torch.ones(mgrid_coords.shape[0], 1) * xr
  yr = torch.ones(mgrid_coords.shape[0], 1) * yr
  thetar = torch.ones(mgrid_coords.shape[0], 1) * zr
  thetar = thetar / (1 * math.pi)

  xh = torch.ones(mgrid_coords.shape[0], 1) * xh
  yh = torch.ones(mgrid_coords.shape[0], 1) * yh
  coords = torch.cat((time_coords, xr, yr, thetar, xh, yh), dim=1) 
  model_in = {'coords': coords.cuda()}
  model_out = model(model_in)['model_out']

  model_out = model_out.detach().cpu().numpy()
  model_out = model_out[0,0]
  # Unnormalize the value function
  norm_to = 0.02
  mean = 0.25
  var = 0.5
  model_out = (model_out*var/norm_to) + mean 
  return model_out

def optUHu(x_hu, y_hu, p3, p4):
  u_star = torch.atan2(goal[1]-y_hu, goal[0]-x_hu) # angle directly to goal
  beta = p*beta1 + (1-p)*beta2
  spread = truncnorm_ppf(0.95,torch.tensor(-np.pi),torch.tensor(np.pi), loc = 0, scale = beta)
  umin = u_star - spread
  umax = u_star + spread
  inner = torch.where(spread < np.pi/2, True, False)
  # control that maximizes the hamiltonians
  control1 = torch.atan2(torch.tensor(p4), torch.tensor(p3))
  control2 = torch.where(control1 > 0, control1 - np.pi, control1 + np.pi)
  
  
  offset1 = torch.where(pointDist(control1,umin,umax), umin, umax)
  control1 = torch.where(arcpoint(control1,umin,umax,inner = inner), control1.double(), offset1.double())


  offset2 = torch.where(pointDist(control2,umin,umax), umin,umax)
  control2 = torch.where(arcpoint(control2,umin,umax,inner = inner), control2.double(), offset2.double())

  ham1 = p3*vel*(torch.cos(control1)) + p4*vel*(torch.sin(control1))
  ham2 = p3*vel*(torch.cos(control2)) + p4*vel*(torch.sin(control2))
  if ham1 < ham2:
    return control1
  else:
    return control2


filenames = []

fig = plt.figure()
plt.scatter(np.array(xro[0]), np.array(xro[1]))
plt.scatter(np.array(xhu[0]), np.array(xhu[1]))
plt.xlim([-1,1])
plt.ylim([-1,1])
fig.savefig('PE_game/roHuPE_t=%0.2f.png' % (t))
filenames.append('PE_game/roHuPE_t=%0.2f.png' % (t))


while t<T:
  time_coords = torch.ones(mgrid_coords.shape[0], 1) * 1. 
  val_state = value(xro[0],xro[1],xro[2],xhu[0],xhu[1])
  if (np.linalg.norm(np.array(xro)[0:2]- np.array(xhu)[0:2]) - 0.2) < 0.001:
    break

  p0 = (value(xro[0] + delta, xro[1], xro[2], xhu[0], xhu[1]) - value(xro[0] - delta,xro[1],xro[2],xhu[0],xhu[1]))/ (2*delta)
  p1 = (value(xro[0] ,xro[1] + delta, xro[2], xhu[0], xhu[1]) - value(xro[0] ,xro[1]- delta,xro[2],xhu[0],xhu[1]))/ (2*delta)
  p2 = (value(xro[0] ,xro[1], xro[2] + delta, xhu[0], xhu[1]) - value(xro[0] ,xro[1],xro[2]- delta,xhu[0],xhu[1]))/ (2*delta)
  p3 = (value(xro[0] ,xro[1], xro[2], xhu[0] + delta, xhu[1]) - value(xro[0] ,xro[1],xro[2],xhu[0]- delta,xhu[1]))/ (2*delta)
  p4 = (value(xro[0] ,xro[1], xro[2], xhu[0], xhu[1] + delta) - value(xro[0] ,xro[1],xro[2],xhu[0],xhu[1]- delta))/ (2*delta)

  if p2<0:
    ur = -omega_max
  else:
    ur = omega_max

  uh = optUHu(xhu[0], xhu[1], p3, p4)

  xro[0] = xro[0] + vel*np.cos(xro[2])*dt
  xro[1] = xro[1] + vel*np.sin(xro[2])*dt
  xro[2] = xro[2] + ur*dt
  xhu[0] = xhu[0] + vel*np.cos(uh)*dt
  xhu[1] = xhu[1] + vel*np.sin(uh)*dt

  xro_t.append(xro[0])
  yro_t.append(xro[1])
  xhu_t.append(xhu[0])
  yhu_t.append(xhu[1])
  t = t + dt
  fig = plt.figure()
  plt.scatter(np.array(xro[0]), np.array(xro[1]), label = 'Robot Position')
  plt.scatter(np.array(xhu[0]), np.array(xhu[1]), label = 'Human Position')
  plt.scatter(0,0, label='Human goal')
  plt.legend()
  plt.xlim([-1,1])
  plt.ylim([-1,1])
  fig.savefig('PE_game/roHuPE_t=%0.2f.png' % (t))
  filenames.append('PE_game/roHuPE_t=%0.2f.png' % (t))

images = []
for filename in filenames:
  images.append(imageio.imread(filename))
imageio.mimsave('movie.gif', images)




'''
# Start plotting the results
for i in range(num_times):
  time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]
  xh_coords = torch.ones(mgrid_coords.shape[0], 1) * 0.
  yh_coords = torch.ones(mgrid_coords.shape[0], 1) * 0.
  
  for j in range(num_thetas):
    theta_coords = torch.ones(mgrid_coords.shape[0], 1) * thetas[j]
    theta_coords = theta_coords / (1 * math.pi)

    coords = torch.cat((time_coords, xh_coords, yh_coords, theta_coords, mgrid_coords), dim=1) 
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
    ax = fig.add_subplot(num_times, num_thetas, (j+1) + i*num_thetas)
    ax.set_title('t = %0.2f, theta = %0.2f' % (times[i], thetas[j]))
    s = ax.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
    print(model_out.T)
    plt.scatter(0, 0)
    fig.colorbar(s) 
'''