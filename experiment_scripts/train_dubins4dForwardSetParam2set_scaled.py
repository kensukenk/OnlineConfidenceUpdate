# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

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
p.add_argument('--collisionR', type=float, default=0.2, required=False, help='Collision radisu between vehicles')

p.add_argument('--minWith', type=str, default='target', required=False, choices=['none', 'zero', 'target'], help='BRS vs BRT computation')

p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--diffModel', action='store_true', default=False, required=False, help='Should we train the difference model instead.')
p.add_argument('--time_norm_mode', type=str, default='target', required=False, choices=['none', 'scale_ham', 'scale_PDE'])

p.add_argument('--periodic_boundary', action='store_true', default=False, required=False, help='Impose the periodic boundary condition.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--pretrain', action='store_true', default=False, required=False, help='Pretrain dirichlet conditions')
p.add_argument('--adjust_relative_grads', action='store_true', default=False, required=False, help='Adjust relative gradients of the loss function.')
p.add_argument('--diffModel_mode', type=str, default='mode1', required=False, choices=['mode1', 'mode2'], help='BRS vs BRT computation')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
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

model = modules.SingleBVPNet(in_features=14, out_features=1, type=opt.model, mode=opt.mode,
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
model.cuda()

# Define the loss
loss_fn = loss_functions.initialize_hji_dubins4d_forward_param(dataset, opt.minWith, opt.diffModel_mode)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Validation function
def val_fn(model, ckpt_dir, epoch):
  # Normalization coefficients
  alpha = dataset.alpha
  beta = dataset.beta

  # Time values at which the function needs to be plotted
  times = [0., 0.1*opt.tMax, 0.25*opt.tMax, 0.5*opt.tMax, 0.75*opt.tMax, opt.tMax-0.1]
  num_times = len(times)

  # Velocity and theta

  


  # Parameter slices to be plotted
  aMin1 = [ 0, 0.22]
  aMax1 = [0, 1.22]

  oMin1 = [ 0, 0.329*alpha['time']]
  oMax1 = [ 0, 0.6*alpha['time']]

  aMin2 = [ 0, -8.04]
  aMax2 = [0, 9.69]
  oMin2 = [ 0, -0.09*alpha['time']]
  oMax2 = [0, 0.755*alpha['time']]

  #startX = [-1.5, -1.5, 0.0, 0.0]
  #startY = [-1.5, -1.5, 0.0, 0.0]
  start_v = 1.62/alpha['time']
  #start_th = 1.22

  
  num_params = len(aMin1)

  # Create a figure
  fig = plt.figure(figsize=(5*num_params, 5*num_times))

  # Get the meshgrid in the (x, y) coordinate
  sidelen = 100
  sidelen = (40,40, 25, 25)
  mgrid_coords = dataio.get_mgrid(sidelen, dim = 4)

  # Start plotting the results
  for i in range(num_times):
    time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]

    for j in range(num_params):
      # State coords 
      coords = torch.cat((time_coords, mgrid_coords), dim=1) 


      # Initial position coords
      #startX_coords = (torch.ones(mgrid_coords.shape[0], 1) * startX[j] - beta['x'])/alpha['x']
      #startY_coords = (torch.ones(mgrid_coords.shape[0], 1) * startY[j] - beta['y'])/alpha['y']
      #startTheta_coords = (torch.ones(mgrid_coords.shape[0], 1) * start_th - beta['th'])/alpha['th']
      startV_coords = (torch.ones(mgrid_coords.shape[0], 1) * start_v - beta['v'])/alpha['v']
      coords = torch.cat((coords, startV_coords), dim=1) 


      # Initial control bounds
      aMin1_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMin1[j] - beta['a'])/alpha['a']
      aMax1_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMax1[j] - beta['a'])/alpha['a']
      oMin1_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMin1[j] - beta['o'])/alpha['o']
      oMax1_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMax1[j] - beta['o'])/alpha['o']
      coords = torch.cat((coords, aMin1_coords, aMax1_coords, oMin1_coords, oMax1_coords), dim=1) 

      # Final control bounds
      aMin2_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMin2[j] - beta['a'])/alpha['a']
      aMax2_coords = (torch.ones(mgrid_coords.shape[0], 1) * aMax2[j] - beta['a'])/alpha['a']
      oMin2_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMin2[j] - beta['o'])/alpha['o']
      oMax2_coords = (torch.ones(mgrid_coords.shape[0], 1) * oMax2[j] - beta['o'])/alpha['o']
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


      # Plot the zero level sets
      model_out = (model_out <= 0.001)*1.

      # Plot the actual data
      ax = fig.add_subplot(num_times, num_params, (j+1) + i*num_params)
      ax.set_title('t = %0.2f' % (times[i]))
      s = ax.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-alpha['x'], alpha['x'], -alpha['y'], alpha['y']), aspect=(alpha['x']/alpha['y']), vmin=-1., vmax=1.)
      fig.colorbar(s) 
      ax.set_aspect('equal')

  fig.savefig(os.path.join(ckpt_dir, 'BRS_validation_plot_epoch_%04d.png' % epoch))
  


training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, clip_grad=opt.clip_grad,
               use_lbfgs=opt.use_lbfgs, validation_fn=val_fn, start_epoch=opt.checkpoint_toload,
               adjust_relative_grads=opt.adjust_relative_grads)
