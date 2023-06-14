import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from helper import *
# Enable import from parent package

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap

import dataio, utils, training, loss_functions, modules

import math
from torch.utils.data import DataLoader
import configargparse


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')


# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=1.0, required=False, help='End time of the simulation')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--num_nl', type=int, default=512, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--pretrain_iters', type=int, default=2000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False, help='Defines the initial time for the curriculul training')
p.add_argument('--counter_end', type=int, default=-1, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=1000, required=False, help='Number of source samples at each time step')

p.add_argument('--collisionR', type=float, default=0.17, required=False, help='Collision radisu between vehicles')
p.add_argument('--minWith', type=str, default='none', required=False, choices=['none', 'zero', 'target'], help='BRS vs BRT computation')

p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--pretrain', action='store_true', default=False, required=False, help='Pretrain dirichlet conditions')

p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
p.add_argument('--periodic_boundary', action='store_true', default=False, required=False, help='Impose the periodic boundary condition.')
p.add_argument('--diffModel', action='store_true', default=True, required=False, help='Should we train the difference model instead.')
p.add_argument('--diffModel_mode', type=str, default='mode2', required=False, choices=['mode1', 'mode2'], help='BRS vs BRT computation')
p.add_argument('--adjust_relative_grads', action='store_true', default=True, required=False, help='Adjust relative gradients of the loss function.')

opt = p.parse_args()

# Set the source coordinates for the target set and the obstacle sets
source_coords = [0., 0., 0.]
if opt.counter_start == -1:
  opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
  opt.counter_end = opt.num_epochs

dataset = dataio.ReachabilityDubins4DForwardParam2SetScaled(numpoints=65000, collisionR=opt.collisionR, 
                                           pretrain=opt.pretrain, tMin=opt.tMin,
                                          tMax=opt.tMax, counter_start=opt.counter_start, counter_end=opt.counter_end,
                                          pretrain_iters=opt.pretrain_iters, 
                                          num_src_samples=opt.num_src_samples, periodic_boundary = opt.periodic_boundary, diffModel=opt.diffModel)

dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# in_features = num states + 1 (for time) + num_params
# t, x,y, x0,y0, umin1, umax1,
model = modules.SingleBVPNet(in_features=14, out_features=1, type=opt.model, mode=opt.mode,
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)

model.cuda()
model.load_state_dict(torch.load('../logs/dubins4dParamFRS_pre40_src12_epo150_rad0017_2set_adjgrad_scaled_time/checkpoints/model_final.pth'))

# Define the loss
loss_fn = loss_functions.initialize_hji_dubins4d_forward_param(dataset, opt.minWith, opt.diffModel_mode)

# Normalization coefficients
alpha = dataset.alpha
beta = dataset.beta

# Visualize the FRT 
#def visGraph(mode, belief, start_v, aMin1, aMax1, oMin1, oMax1, aMin2, aMax2, oMin2, oMax2,humPred, humState, t):
def visGraph(mode, start_v, bounds, humHist, humPred, humState, robStates, robPreds, robPasts, t, beta_interp):
  aMin1_nobel, aMax1_nobel, oMin1_nobel, oMax1_nobel, aMin2_nobel, aMax2_nobel, oMin2_nobel, oMax2_nobel = bounds[0]
  aMin1_bel, aMax1_bel, oMin1_bel, oMax1_bel, aMin2_bel, aMax2_bel, oMin2_bel, oMax2_bel = bounds[1]

  aMin1 = [aMin1_nobel, aMin1_bel]
  aMax1 = [aMax1_nobel, aMax1_bel]
  # accounting for time scaling
  oMin1 = [oMin1_nobel*alpha['time'], oMin1_bel*alpha['time']]
  oMax1 = [oMax1_nobel*alpha['time'], oMax1_bel*alpha['time']]
  aMin2 = [aMin2_nobel, aMin2_bel]
  aMax2 = [aMax2_nobel, aMax2_bel]
  oMin2 = [oMin2_nobel*alpha['time'],oMin2_bel*alpha['time']]
  oMax2 = [oMax2_nobel*alpha['time'], oMax2_bel*alpha['time']]
  start_v = [start_v/alpha['time']]

  num_params = 1
  # Time values at which the function needs to be plotted
  times = [opt.tMax]
  num_times = len(times)
  # Create a figure
  plt.rcParams['axes.grid'] = False
  plt.axis('off')
  plt.grid(b=None)

  fig = plt.figure(figsize=(10*num_params, 5*num_times))

  # Get the meshgrid in the (x, y) coordinate
  sidelen = (95, 95, 75)
  mgrid_coords = dataio.get_mgrid(sidelen, dim = 3)

  # minimize memory usage by doing velocity coordinate sequentially
  vcoords = np.linspace(-1, 1, 50)
  # Start plotting the results
  for i in range(num_times):
    time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]
    for j in range(num_params):
      for vel in vcoords:
        # State coords 
        coords = torch.cat((time_coords, mgrid_coords), dim=1) 

        vel_coords = torch.ones(mgrid_coords.shape[0], 1) * vel
        coords = torch.cat((coords, vel_coords), dim = 1)

        startV_coords = (torch.ones(mgrid_coords.shape[0], 1) * start_v[j] - beta['v'])/alpha['v']
        coords = torch.cat((coords, startV_coords), dim=1) 


        # Initial control bounds
        aMin1_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMin1[j] - beta['a'])/alpha['a']
        aMax1_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMax1[j] - beta['a'])/alpha['a']
        oMin1_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMin1[j] - beta['o'])/alpha['o']
        oMax1_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMax1[j] - beta['o'])/alpha['o']

        bel_aMin1_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMin1[j+1] - beta['a'])/alpha['a']
        bel_aMax1_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMax1[j+1] - beta['a'])/alpha['a']
        bel_oMin1_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMin1[j+1] - beta['o'])/alpha['o']
        bel_oMax1_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMax1[j+1] - beta['o'])/alpha['o']
        nobel_coords = torch.cat((coords, aMin1_coords, aMax1_coords, oMin1_coords, oMax1_coords), dim=1) 
        bel_coords = torch.cat((coords, bel_aMin1_coords, bel_aMax1_coords, bel_oMin1_coords, bel_oMax1_coords), dim=1) 

        # Final control bounds
        aMin2_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMin2[j] - beta['a'])/alpha['a']
        aMax2_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMax2[j] - beta['a'])/alpha['a']
        oMin2_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMin2[j] - beta['o'])/alpha['o']
        oMax2_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMax2[j] - beta['o'])/alpha['o']

        bel_aMin2_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMin2[j+1] - beta['a'])/alpha['a']
        bel_aMax2_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMax2[j+1] - beta['a'])/alpha['a']
        bel_oMin2_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMin2[j+1] - beta['o'])/alpha['o']
        bel_oMax2_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMax2[j+1] - beta['o'])/alpha['o']
        nobel_coords = torch.cat((nobel_coords, aMin2_coords, aMax2_coords, oMin2_coords, oMax2_coords), dim=1) 
        bel_coords = torch.cat((bel_coords, bel_aMin2_coords, bel_aMax2_coords, bel_oMin2_coords, bel_oMax2_coords), dim=1) 

        FRTs = []

        coords = [nobel_coords, bel_coords]
        for k in range(2):
          model_in = {'coords': coords[k].cuda()}
          model_out = model(model_in)['model_out']

          # Detatch model ouput and reshape
          model_out = model_out.detach().cpu().numpy()
          model_out = model_out.reshape(sidelen)

          # Unnormalize the value function
          model_out = (model_out*dataset.var/dataset.norm_to) + dataset.mean 

          if opt.diffModel:
            lx = dataset.compute_IC(coords[k][..., 1:])
            lx = lx.detach().cpu().numpy()
            lx = lx.reshape(sidelen)
            if opt.diffModel_mode == 'mode1':
              model_out = model_out + lx
            elif opt.diffModel_mode == 'mode2':
              model_out = model_out + lx - dataset.mean
            else:
              raise NotImplementedError
          model_out = np.min(model_out, axis = -1) # union over theta

          if vel == vcoords[0]:
            if k == 0:
              nobel_FRT_out = model_out
            else:
              bel_FRT_out = model_out
          else:
            if k == 0:
              nobel_FRT_out = np.minimum(nobel_FRT_out, model_out)
            else:
              bel_FRT_out = np.minimum(bel_FRT_out, model_out)
          

      # Plot the zero level sets
      nobel_FRT_out = (nobel_FRT_out <= 0.00001)*1.
      bel_FRT_out = (bel_FRT_out <= 0.00001)*1.
      im = plt.imread("icons/UTurnMap_final.png")
      #im = plt.imread("icons/StopLineMap_final.png")
      ax = fig.add_subplot(121)
   
      im = ax.imshow(im, extent=[-2.4067, 2.4067, -2.4067, 2.4067])
      # Plot the actual data


      # making new colormaps
      nobel_colors = [(1, 1, 1), (119/256, 56/256, 137/256)]  # R -> G -> B
      bel_colors = [(1, 1, 1), (55/256, 138/256, 110/256)]  # R -> G -> B
      n_bin = 2  # Discretizes the interpolation into bins

      bel_cmap = LinearSegmentedColormap.from_list('bel', bel_colors, N=n_bin)
      nobel_cmap = LinearSegmentedColormap.from_list('nobel', nobel_colors, N=n_bin)

      
      scale = 1/9
      xref = -0.805
      yref = -.735
      xcent = 105.245
      ycent = 71.955
      humPredX = humPred[:,0]
      humPredY = humPred[:,1]
      humPredX = scale*(humPredX - xcent) + xref
      humPredY = scale*(humPredY - ycent) + yref

      robPred, robPredConf = robPreds
      robPredX, robPredY = robPred
      robPredX = scale*(np.array(robPredX) - xcent) + xref
      robPredY = scale*(np.array(robPredY) - ycent) + yref

      robPredConfX, robPredConfY = robPredConf
      robPredConfX = scale*(np.array(robPredConfX) - xcent) + xref
      robPredConfY = scale*(np.array(robPredConfY) - ycent) + yref

      robTrajPast, robTrajConfPast = robPasts
      robPastX, robPastY = robTrajPast
      robPastX = scale*(np.array(robPastX) - xcent) + xref
      robPastY = scale*(np.array(robPastY) - ycent) + yref

      robConfPastX, robConfPastY = robTrajConfPast
      robConfPastX = scale*(np.array(robConfPastX) - xcent) + xref
      robConfPastY = scale*(np.array(robConfPastY) - ycent) + yref

      humHistX = humHist[0]
      humHistY = humHist[1]
      humHistX = scale*(humHistX - xcent) + xref
      humHistY = scale*(humHistY - ycent) + yref
      humStartX, humStartY, humStartThet = humState

      robState, robConfState = robStates      
      robStartX, robStartY, robStartThet = robState
      robConfStartX, robConfStartY, robConfStartThet = robConfState

      humStartX = scale*(humStartX - xcent) + xref
      humStartY = scale*(humStartY - ycent) + yref
      robStartX = scale*(robStartX - xcent) + xref
      robStartY = scale*(robStartY - ycent) + yref
      robConfStartX = scale*(robConfStartX - xcent) + xref
      robConfStartY = scale*(robConfStartY - ycent) + yref
      ax.scatter(humPredX, humPredY, s=5, c='skyblue', label = 'Predictions')
      ax.plot(humPredX, humPredY, c='skyblue')
      ax.scatter(humHistX, humHistY, marker = '*', s=8, c='b', label = 'Human History')
      ax.plot(humHistX, humHistY, c='b')

      ax.scatter(robPredX, robPredY,  s=8,  color='gold', label = 'Trajectory Without Confidence')
      ax.scatter(robPredConfX, robPredConfY, s=8, color='red', label = 'Confidence-adjusted Trajectory')
      ax.scatter(robPastX, robPastY, marker = '*', s=8, alpha = 0.5, c='grey', label = 'Robot History Without Confidence')
      ax.scatter(robConfPastX, robConfPastY,marker = '*', s=8, c='rosybrown', label = 'Confidence-adjusted Robot History')
      ax.plot(robPredX, robPredY, color='gold' )
      ax.plot(robPredConfX, robPredConfY, color='red' )
      ax.plot(robPastX, robPastY, alpha = 0.5, c='grey' )
      ax.plot(robConfPastX, robConfPastY, c='rosybrown')


      s_bel = ax.imshow(bel_FRT_out.T, cmap=bel_cmap, alpha = bel_FRT_out.T*0.7, origin='lower', extent=(-alpha['x'], alpha['x'], -alpha['y'],alpha['y']), aspect=(alpha['x']/alpha['y']), vmin=-1., vmax=1.)
      s_nobel = ax.imshow(nobel_FRT_out.T, cmap=nobel_cmap, alpha = nobel_FRT_out.T*0.7, origin='lower', extent=(-alpha['x'], alpha['x'], -alpha['y'],alpha['y']), aspect=(alpha['x']/alpha['y']), vmin=-1., vmax=1.)
      blue_car = plt.imread("icons/Car TOP_VIEW 375397.png")
      red_car = plt.imread("icons/Car TOP_VIEW F05F78.png")

      s_ego = ax.imshow(red_car, alpha=0.6, extent=[-0.5/2, 0.5/2, -0.5/4, 0.5/4])
      s_conf_ego = ax.imshow(red_car, extent=[-0.5/2, 0.5/2, -0.5/4, 0.5/4])
      s_car = ax.imshow(blue_car, extent=[-0.5/2, 0.5/2, -0.5/4, 0.5/4])

      
      trans_data = mtransforms.Affine2D().rotate_around(-1,0,humStartThet).translate(humStartX-(-1), humStartY-0) + ax.transData
      trans_data_car = mtransforms.Affine2D().rotate_around(0,0,humStartThet).translate(humStartX, humStartY) + ax.transData
      trans_data_ego = mtransforms.Affine2D().rotate_around(0,0,robStartThet).translate(robStartX, robStartY) + ax.transData
      trans_data_conf_ego = mtransforms.Affine2D().rotate_around(0,0,robConfStartThet).translate(robConfStartX, robConfStartY) + ax.transData
      s_bel.set_transform(trans_data)
      s_nobel.set_transform(trans_data)
      s_ego.set_transform(trans_data_ego)
      s_conf_ego.set_transform(trans_data_conf_ego)
      s_car.set_transform(trans_data_car)

      ax.set_aspect('equal')
      ax.set_xlim([-2.4067, 2.4067])
      ax.set_ylim([-2.4067, 2.4067]) 

      ax.legend(fontsize="5", loc ="upper right")
      ax.axis('off')

      ax2 = fig.add_subplot(122)
      ax2.scatter((t)/2, beta_interp[t])
      ax2.plot(np.arange(t+1)/2, beta_interp[:t+1])
      ax2.set_ylim([0,1.1])
      ax2.set_xlim([-0.1,11])
      ax2.set_title('Model Confidence')
      ax2.set_xlabel('Time(s)')
      ax2.set_ylabel('Belief')
   

  
  fig.savefig(mode + str(t)+'figs.png', dpi=400)





# Check the value of a given state
def queryFRT(xCheck, yCheck, start_v, aMin1, aMax1, oMin1, oMax1, aMin2, aMax2, oMin2, oMax2):
  # Get the meshgrid in the (x, y) coordinate
  sidelen = (2, 2, 40, 40)
  mgrid_coords = dataio.get_mgrid(sidelen, dim = 4)
  
  # account for time scaling
  xCheck = xCheck/9.0
  yCheck = yCheck/9.0
  start_v = start_v/alpha['time']
  oMin1 = oMin1*alpha['time']
  oMax1 = oMax1*alpha['time']
  oMin2 = oMin2*alpha['time']
  oMax2 = oMax2*alpha['time']
  
  # check final value FRT
  time_coords = torch.ones(mgrid_coords.shape[0], 1) * 1.0
  # State coords 
  mgrid_coords[..., 0] = (torch.ones(mgrid_coords.shape[0], ) * xCheck - beta['x'])/alpha['x']
  mgrid_coords[..., 1] = (torch.ones(mgrid_coords.shape[0], ) * yCheck - beta['y'])/alpha['y']
  coords = torch.cat((time_coords, mgrid_coords), dim=1) 

  startV_coords = (torch.ones(mgrid_coords.shape[0], 1) * start_v - beta['v'])/alpha['v']
  coords = torch.cat((coords, startV_coords), dim=1) 

  # Initial control bounds
  aMin1_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMin1 - beta['a'])/alpha['a']
  aMax1_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMax1 - beta['a'])/alpha['a']
  oMin1_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMin1 - beta['o'])/alpha['o']
  oMax1_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMax1 - beta['o'])/alpha['o']
  coords = torch.cat((coords, aMin1_coords, aMax1_coords, oMin1_coords, oMax1_coords), dim=1) 

  # Final control bounds
  aMin2_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMin2 - beta['a'])/alpha['a']
  aMax2_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMax2 - beta['a'])/alpha['a']
  oMin2_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMin2 - beta['o'])/alpha['o']
  oMax2_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMax2 - beta['o'])/alpha['o']
  coords = torch.cat((coords, aMin2_coords, aMax2_coords, oMin2_coords, oMax2_coords), dim=1) 

  model_in = {'coords': coords.cuda()}
  model_out = model(model_in)['model_out']

  # Detatch model ouput and reshape
  model_out = model_out.detach().cpu().numpy()
  model_out = model_out.reshape(sidelen)


  # Unnormalize the value function
  model_out = (model_out*dataset.var/dataset.norm_to) + dataset.mean 

  if opt.diffModel:
    lx = dataset.compute_IC(coords[..., 1:])
    lx = lx.detach().cpu().numpy()
    lx = lx.reshape(sidelen)
    if opt.diffModel_mode == 'mode1':
      model_out = model_out + lx
    elif opt.diffModel_mode == 'mode2':
      model_out = model_out + lx - dataset.mean
    else:
      raise NotImplementedError
  

  model_out = np.min(model_out, axis = -1) # union over velocity
  model_out = np.min(model_out, axis = -1) # union over theta

  # value of state
  return model_out[0,0]
      