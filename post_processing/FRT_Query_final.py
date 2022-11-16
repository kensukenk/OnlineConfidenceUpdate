import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('../Trajectron-plus-plus/trajectron')
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import numpy as np
import torch
import dill
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from helper import *
from numpy.linalg import lstsq
from scipy.stats import multivariate_normal, norm

# Enable import from parent package

import matplotlib.pyplot as plt


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
def visGraph(mode, belief, start_v, aMin1, aMax1, oMin1, oMax1, aMin2, aMax2, oMin2, oMax2):
  aMin1 = [aMin1]
  aMax1 = [aMax1]
  # accounting for time scaling
  oMin1 = [ oMin1*alpha['time']]
  oMax1 = [ oMax1*alpha['time'] ]
  aMin2 = [aMin2]
  aMax2 = [aMax2]
  oMin2 = [oMin2*alpha['time']]
  oMax2 = [oMax2*alpha['time']]
  start_v = [start_v/alpha['time']]

  num_params = len(aMin1)
  # Time values at which the function needs to be plotted
  times = [opt.tMax]
  num_times = len(times)
  # Create a figure
  plt.rcParams['axes.grid'] = False
  plt.axis('off')
  plt.grid(b=None)
  fig = plt.figure(figsize=(5*num_params, 5*num_times))

  # Get the meshgrid in the (x, y) coordinate
  sidelen = (100, 100, 100)
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
        model_out = np.min(model_out, axis = -1) # union over theta

        if vel == vcoords[0]:
          FRT_out = model_out
        else:
          FRT_out = np.minimum(FRT_out, model_out)

      # Plot the zero level sets
      FRT_out = (FRT_out <= 0.001)*1.

      # Plot the actual data
      ax = fig.add_subplot(num_times, num_params, (j+1) + i*num_params)
      ax.set_title('t = %0.2f' % (times[i]))
      s = ax.imshow(FRT_out.T, cmap='bwr', origin='lower', extent=(-alpha['x'], alpha['x'], -alpha['y'],alpha['y']), aspect=(alpha['x']/alpha['y']), vmin=-1., vmax=1.)
      ax.set_aspect('equal')
      plt.scatter(-1,0, s=5)
  fig.savefig(mode + belief +'.png')

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
      

# Least squares approximation of controls
def least_squares(controls):
    b1 = np.array(controls)
    numT = np.size(controls) / 2 # timestep is 0.5s long
    tsteps = np.arange(0, numT, 0.5)
    
    # assemble matrix A
    A = np.vstack([tsteps, np.ones(len(tsteps))]).T

    # turn y into a column vector
    b1 = b1[:, np.newaxis]
    
    sol = lstsq(A, b1)[0]
    M = sol[0]
    b = sol[1]
    return M, b

# transform global coordinates to coordinates relative to the human
def relativeCoords(x1,y1,z1,x2,y2,z2):
    x_rel, y_rel = centeredPos(x1,y1,x2,y2)
    r, theta = toPolar(x_rel, y_rel)
    theta_pos = theta - z1
    x_pos, y_pos = toCartesian(r, theta_pos)
    theta_rel = anglehelper(z2 - z1)
    return x_pos, y_pos, theta_rel

# center around x1, y1  
def centeredPos(x1,y1,x2,y2):
    x_rel = x2 - x1
    y_rel = y2 - y1
    return x_rel, y_rel

def toPolar(x,y):
    r = np.sqrt(np.power(x,2) + np.power(y,2))
    theta = np.arctan2(y,x)
    return r, theta
def toCartesian(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y
def anglehelper(theta):
    theta = np.where(theta < -np.pi, theta + 2*np.pi, theta)
    theta = np.where(theta > np.pi, theta - 2*np.pi, theta)
    return theta


def main(mode):
  nuScenes_data_path = './v1.0' # Data Path to nuScenes data set 
  nuScenes_devkit_path = './devkit/python-sdk/'
  sys.path.append(nuScenes_devkit_path)

  # NuScenes parameters relevant to the scene
  if mode == 'uturn':
    with open('../Trajectron-plus-plus/experiments/processed/nuScenes_uturntest_full.pkl', 'rb') as f:
      eval_env = dill.load(f, encoding='latin1')
      eval_scenes = eval_env.scenes
      hum_idx = 1
      rob_idx = 0
      scene_idx = 1
      frames = torch.arange(3,35, 1)

  elif mode == 'stop':
    with open('../Trajectron-plus-plus/experiments/processed/nuScenes_ICRAtest_full.pkl', 'rb') as f:
      eval_env = dill.load(f, encoding='latin1')
      eval_scenes = eval_env.scenes
      hum_idx = 1
      rob_idx = -1
      scene_idx = 0
      frames = torch.arange(1,19, 1)

  else:
    raise Exception("Only U-turn or Stop sign scenarios are allowed.")

  # probability mass bounds
  prob_min = 0.465
  prob_max = 0.535
  # lower bound of beta
  beta = 0.03


  ph = 6 # prediction horizon is 6 timesteps of 0.5 seconds
  log_dir = './models'

  # load pretrained trajectron++ predictor
  model_dir = os.path.join(log_dir, 'int_ee_me') 
  eval_stg, hyp = load_model(model_dir, eval_env, ts=12)
  scene = eval_scenes[scene_idx]

  # Define ROI in nuScenes Map
  x_min = 773.0
  x_max = 1100.0
  y_min = 1231.0
  y_max = 1510.0

  xrange = x_max - x_min
  yrange = y_max - y_min
  x_min = scene.x_min
  x_max = x_min + xrange
  y_min = scene.y_min
  y_max = y_min + yrange

  # parameters to store
  omin = []
  omin_bel = []
  omax = []
  omax_bel = []
  amin = []
  amin_bel = []
  amax = []
  amax_bel = []
  omean = []
  ostd = []
  ostd_bel = []
  amean = []
  astd = []
  astd_bel = []

  mus = []
  covs = []
  covsBel = []

  humVel = []
  humX = []
  humY = []
  humThet = []

  robVel = []
  robX = []
  robY = []
  robThet = []


  beliefs = [0.5] # Initial prior
  eps = 1e-8 # amount to sample from initial prior (see: Confidence-aware motion prediction for real-time collision avoidance, Fridovich-Keil)

  # storing actions, beliefs, and prediction mean and covariances
  with torch.no_grad():
      for tstep in frames:
          tstep = int(tstep) + 1
          timestep = np.array([tstep])
          
          predictions_mm_actions, predictions_mm_sig = eval_stg.predict_actions(scene,
                                            timestep,
                                            ph,
                                            num_samples=1,
                                            z_mode=True,
                                            gmm_mode=True)
          
          actkeys0 = [*predictions_mm_actions[tstep]]
          pred_agent_act = predictions_mm_actions[tstep][actkeys0[1]]
          pred_agent_act = np.array(pred_agent_act[0,0,:,:])
          
          sigkeys0 = [*predictions_mm_sig[tstep]]
          pred_agent_sig = predictions_mm_sig[tstep][sigkeys0[1]]
          pred_agent_sig = np.array(pred_agent_sig[0,0,:,0,:,:])

          steering = scene.nodes[hum_idx].data.data[tstep][9]
          
          axHum = scene.nodes[hum_idx].data.data[tstep][4]
          ayHum = scene.nodes[hum_idx].data.data[tstep][5]
          vxHum = scene.nodes[hum_idx].data.data[tstep][2]
          vyHum = scene.nodes[hum_idx].data.data[tstep][3]
          ax2 = ((vxHum*axHum + vyHum*ayHum) / (vxHum*vxHum + vyHum*vyHum))*vxHum
          ay2 = ((vxHum*axHum + vyHum*ayHum) / (vxHum*vxHum + vyHum*vyHum))*vyHum
          vel_ang = np.arctan2(vyHum,vxHum)
          velHum = np.sqrt(vxHum*vxHum + vyHum*vyHum)
          humX.append(scene.nodes[hum_idx].data.data[tstep][0])
          humY.append(scene.nodes[hum_idx].data.data[tstep][1])
          humVel.append(velHum)
          humThet.append(vel_ang)
          
          vxRob = scene.nodes[rob_idx].data.data[tstep][2]
          vyRob = scene.nodes[rob_idx].data.data[tstep][3]
          vel_angRob = np.arctan2(vyRob,vxRob)
          velRob = np.sqrt(vxRob*vxRob + vyRob*vyRob)
          robX.append(scene.nodes[rob_idx].data.data[tstep][0])
          robY.append(scene.nodes[rob_idx].data.data[tstep][1])
          robVel.append(velRob)
          robThet.append(vel_angRob)
     
          acc_ang = np.arctan2(ay2,ax2)
          if np.abs(vel_ang - acc_ang) > 1:
              accel = -np.sqrt(ax2*ax2 + ay2*ay2)
          else:
              accel = np.sqrt(ax2*ax2 + ay2*ay2)


          var = multivariate_normal(mean=pred_agent_act[0], cov=pred_agent_sig[0])
          prob1 = var.pdf([steering, accel])
          var2 = multivariate_normal(mean=pred_agent_act[0], cov=(1/beta)*pred_agent_sig[0])
          prob2 = var2.pdf([steering, accel])
          
          prior = (1-eps)*beliefs[-1] + eps*beliefs[0]
          beliefs.append(prior*prob1 / (prior*prob1 + (1 - prior)*prob2))
        
          newMu = pred_agent_act
          
          newCovBel = (beliefs[-1] + (1/beta)*(1-beliefs[-1])) * pred_agent_sig
          newCov = pred_agent_sig
          
          mus.append(newMu)
          covs.append(newCov)
          covsBel.append(newCovBel)

  humX = np.array(humX)
  humY = np.array(humY)
  humThet = np.array(humThet)
  humVel = np.array(humVel)

  robX = np.array(robX) 
  robY = np.array(robY) 
  robThet = np.array(robThet)
  robVel = np.array(robVel)
  
  newMu = mus[-1]
  newCovBel = covsBel[-1]
  newCov = covs[-1]
  aMax1 = []
  oMax1 = []
  aMax1Bel = []
  oMax1Bel = []
  aMin1 = []
  oMin1 = []
  aMin1Bel = []
  oMin1Bel = []
  aMax2 = []
  oMax2 = []
  aMax2Bel = []
  oMax2Bel = []
  aMin2 = []
  oMin2 = []
  aMin2Bel = []
  oMin2Bel = []

  # storing least squares approximation of truncated predicted controls
  for t in range(len(frames)):
      newMu = mus[t]
      newCov = covs[t]
      newCovBel = covsBel[t]
      omin = []
      omin_bel = []
      omax = []
      omax_bel = []
      amin = []
      amin_bel = []
      amax = []
      amax_bel = []
      omean = []
      ostd = []
      ostd_bel = []
      amean = []
      astd = []
      astd_bel = []
      
      for i in range(ph):
          mean_w = newMu[i,0]
          mean_a = newMu[i,1]
          std_w_bel = newCovBel[i,0,0]
          std_a_bel = newCovBel[i,1,1]
          std_w = newCov[i,0,0]
          std_a = newCov[i,1,1]
          omean.append(mean_w)
          ostd.append(std_w)
          ostd_bel.append(std_w_bel)
          amean.append(mean_a)
          astd.append(std_a)
          astd_bel.append(std_a_bel)
          omin.append(norm.ppf(prob_min, mean_w, std_w))
          omax.append(norm.ppf(prob_max, mean_w, std_w))
          amin.append(norm.ppf(prob_min, mean_a, std_a))
          amax.append(norm.ppf(prob_max, mean_a, std_a))
          omin_bel.append(norm.ppf(prob_min, mean_w, std_w_bel))
          omax_bel.append(norm.ppf(prob_max, mean_w, std_w_bel))
          amin_bel.append(norm.ppf(prob_min, mean_a, std_a_bel))
          amax_bel.append(norm.ppf(prob_max, mean_a, std_a_bel))

      (A_omax, b_omax) = least_squares(omax)
      (A_omin, b_omin) = least_squares(omin)
      (A_amax, b_amax) = least_squares(amax)
      (A_amin, b_amin) = least_squares(amin)
      (A_omaxBel, b_omaxBel) = least_squares(omax_bel)
      (A_ominBel, b_ominBel) = least_squares(omin_bel)
      (A_amaxBel, b_amaxBel) = least_squares(amax_bel)
      (A_aminBel, b_aminBel) = least_squares(amin_bel)

      oMax1.append((b_omax)[0])
      oMin1.append((b_omin)[0])
      aMax1.append((b_amax)[0])
      aMin1.append((b_amin)[0])
      oMax1Bel.append((b_omaxBel)[0])
      oMin1Bel.append((b_ominBel)[0])
      aMax1Bel.append((b_amaxBel)[0])
      aMin1Bel.append((b_aminBel)[0])
      
      oMax2.append((b_omax + A_omax*3)[0])
      oMin2.append((b_omin + A_omin*3)[0])
      aMax2.append((b_amax + A_amax*3)[0])
      aMin2.append((b_amin + A_amin*3)[0])
      oMax2Bel.append((b_omaxBel + A_omaxBel*3)[0])
      oMin2Bel.append((b_ominBel + A_ominBel*3)[0])
      aMax2Bel.append((b_amaxBel + A_amaxBel*3)[0])
      aMin2Bel.append((b_aminBel + A_aminBel*3)[0])
  collision = False
  collisionBel = False

  # scene based starting and stopping parameters
  if mode =='uturn':
    stopX = 115.89
    stopY = 83.7
    egoVel = 21
    egoThet = 2.465
    robStartX = robX[11] - egoVel*np.cos(egoThet)*0.5*12
    robStartY = robY[11] - egoVel*np.sin(egoThet)*0.5*12
  if mode == 'stop':
    stopX = robX[7]
    stopY = robY[7]
    robStartX = robX[0] + 3.5*6.8
    robStartY = robY[0] + 5.5*6.8
    egoThet = robThet[0]
    egoVel = 21

  # check predictions and FRT
  for frame in range(len(frames)):
    # start state that relative position measured from
    humStartX = humX[frame]
    humStartY = humY[frame]
    humStartThet = humThet[frame]
    humStartVel = humVel[frame]
    
    egoStartX = robStartX + egoVel*np.cos(egoThet)*frame*0.5
    egoStartY = robStartY + egoVel*np.sin(egoThet)*frame*0.5
    egoX = robStartX + egoVel*np.cos(egoThet)*frame*0.5
    egoY = robStartY + egoVel*np.sin(egoThet)*frame*0.5    
    
    amin1 = aMin1[frame]
    amax1 = aMax1[frame]
    omin1 = oMin1[frame]
    omax1 = oMax1[frame]
    amin2 = aMin2[frame]
    amax2 = aMax2[frame]
    omin2 = oMin2[frame]
    omax2 = oMax2[frame]

    amin1Bel = aMin1Bel[frame]
    amax1Bel = aMax1Bel[frame]
    omin1Bel = oMin1Bel[frame]
    omax1Bel = oMax1Bel[frame]
    amin2Bel = aMin2Bel[frame]
    amax2Bel = aMax2Bel[frame]
    omin2Bel = oMin2Bel[frame]
    omax2Bel = oMax2Bel[frame]

    if omin1 > omax1:
      omin1, omax1 = omax1, omin1
    if omin2 > omax2:
      omin2, omax2 = omax2, omin2
    if amin1 > amax1:
      amin1, amax1 = amax1, amin1
    if amin2 > amax2:
      amin2, amax2 = amax2, amin2

    if omin1Bel > omax1Bel:
      omin1Bel, omax1Bel = omax1Bel, omin1Bel
    if omin2Bel > omax2Bel:
      omin2Bel, omax2Bel = omax2Bel, omin2Bel
    if amin1Bel > amax1Bel:
      amin1Bel, amax1Bel = amax1Bel, amin1Bel
    if amin2Bel > amax2Bel:
      amin2Bel, amax2Bel = amax2Bel, amin2Bel

    # set turn radius to max 2 rad/s
    if omin1 < -2/3:
      omin1 = -2/3
    if omin2 < -2/3:
      omin2 = -2/3
    if omax1 < -2/3:
      omax1 = -2/3
    if omax2 < -2/3:
      omax2 = -2/3
    if omin1 > 2/3:
      omin1 = 2/3
    if omin2 > 2/3:
      omin2 = 2/3
    if omax1 > 2/3:
      omax1 = 2/3
    if omax2 > 2/3:
      omax2 = 2/3

    if omin1Bel < -2/3:
      omin1Bel = -2/3
    if omin2Bel < -2/3:
      omin2Bel = -2/3
    if omax1Bel < -2/3:
      omax1Bel = -2/3
    if omax2Bel < -2/3:
      omax2Bel = -2/3
    
    if omin1Bel > 2/3:
      omin1Bel = 2/3
    if omin2Bel > 2/3:
      omin2Bel = 2/3
    if omax1Bel > 2/3:
      omax1Bel = 2/3
    if omax2Bel > 2/3:
      omax2Bel = 2/3

    # trajectories
    rob_xTraj = [egoX]
    rob_yTraj = [egoY] 
    rob_zTraj = []
    rob_xTrajStop = []
    rob_yTrajStop = []

    hum_xTrajStop = []
    hum_yTrajStop = []
    hum_zTrajStop = []
    
    egoXOld = None
    egoYOld = None
    egoThetOld = None

    for t in range(ph+1):   
        egoXOld = egoX 
        egoYOld = egoY
        egoThetOld = egoThet

        egoX += egoVel*np.cos(egoThet)*0.5
        egoY += egoVel*np.sin(egoThet)*0.5
        rob_xTraj.append(egoX)         
        rob_yTraj.append(egoY)
        rob_zTraj.append(egoThet)

        # want to check over range of points in the time interval
        egoXRange = np.linspace(egoXOld, egoX, 15)
        egoYRange = np.linspace(egoYOld, egoY, 15)
        egoThetRange = np.linspace(egoThetOld, egoThet, 15)
        
        for i in range(len(egoXRange)):
          relX, relY, _ = relativeCoords(humStartX, humStartY, humStartThet, egoXRange[i], egoYRange[i], egoThetRange[i])
          #print(relX, relY)
          querypoint = queryFRT(relX, relY, humStartVel, amin1, amax1, omin1, omax1, amin2, amax2, omin2, omax2)
          querypoint_bel = queryFRT(relX, relY, humStartVel, amin1Bel, amax1Bel, omin1Bel, omax1Bel, amin2Bel, amax2Bel, omin2Bel, omax2Bel)
        
          if collisionBel == False:
            collisionBel = collisionCheck(querypoint_bel, frame, t, egoStartX, egoStartY, egoThet, egoVel, humX, humY, humThet, stopX, stopY, rob_xTrajStop, rob_yTrajStop, hum_xTrajStop, hum_yTrajStop, hum_zTrajStop)
            
            if collisionBel:
              print("time of detected collision: ", t)
              visGraph(mode, '_bel', humStartVel, amin1Bel, amax1Bel, omin1Bel, omax1Bel, amin2Bel, amax2Bel, omin2Bel, omax2Bel)
              visGraph(mode, '_noBel', humStartVel, amin1, amax1, omin1, omax1, amin2, amax2, omin2, omax2)
      
              #print('no bel controls')
              #print(humStartVel, amin1, amax1, omin1, omax1, amin2, amax2, omin2, omax2)
              #print('bel controls')
              #print(humStartVel, amin1Bel, amax1Bel, omin1Bel, omax1Bel, amin2Bel, amax2Bel, omin2Bel, omax2Bel)
              print('human state')
              print(humStartX)
              print(humStartY)
              print(humStartThet)
              print('rob traj')
              print(rob_xTraj)
              print(rob_yTraj)

          if collision == False:
            collision = collisionCheck(querypoint, frame, t, egoStartX, egoStartY, egoThet, egoVel, humX, humY, humThet, stopX, stopY, rob_xTrajStop, rob_yTrajStop, hum_xTrajStop, hum_yTrajStop, hum_zTrajStop)
            if collision:
              print("time of detected collision: ", t)
              visGraph(mode, '_noBel2', humStartVel, amin1, amax1, omin1, omax1, amin2, amax2, omin2, omax2)
              #print(humStartVel, amin1, amax1, omin1, omax1, amin2, amax2, omin2, omax2)
              print('human state')
              print(humStartX)
              print(humStartY)
              print(humStartThet)
              print('rob traj')
              print(rob_xTraj)
              print(rob_yTraj)

        if collision and collisionBel:
          return('Collision')
  return('safe')


def collisionCheck(queryVal, frame, t, egoStartX, egoStartY, egoThet, egoVel, humX, humY, humThet, stopX, stopY,rob_xTrajStop, rob_yTrajStop, hum_xTrajStop, hum_yTrajStop, hum_zTrajStop):
  if queryVal < 0:
    stopDist = np.sqrt((egoStartX-stopX)**2 + (egoStartY-stopY)**2)
    if stopX > egoStartX:
      stopDist = -stopDist
    closestDist(frame, stopDist, egoStartX, egoStartY, egoThet, egoVel, humX, humY, humThet, rob_xTrajStop, rob_yTrajStop, hum_xTrajStop, hum_yTrajStop, hum_zTrajStop)

    return True
  else:
    return False

def closestDist(frame, stopDist, egoX, egoY, robThet, egoVel, humXs, humYs, humThets, rob_xTrajStop, rob_yTrajStop, hum_xTrajStop, hum_yTrajStop, hum_zTrajStop):
  robTrajX = rob_xTrajStop
  robTrajY = rob_yTrajStop
  humTrajX = hum_xTrajStop
  humTrajY = hum_yTrajStop
  humTrajZ = hum_zTrajStop

  if stopDist <= 0:
    stopDist = 0.01
  closest = np.inf
  closestT = 0
  t = 0
  initFrame = frame

  accel, stopTime = stopAccelTime(egoVel,stopDist)
  print('Time to stop: ', stopTime, " seconds")
  print('Decelleration applied: ', accel, " m/s^2")
  while frame < len(humXs):
    if t < stopTime:
      dist = egoVel*t + 0.5*(accel)*(t*t)
      robX = egoX + dist*np.cos(robThet)
      robY = egoY + dist*np.sin(robThet)
      robTrajX.append(robX)
      robTrajY.append(robY)
    else:
      dist = egoVel*stopTime + 0.5*(accel)*(stopTime*stopTime)
      robX = egoX + dist*np.cos(robThet)
      robY = egoY + dist*np.sin(robThet)
      robTrajX.append(robX)
      robTrajY.append(robY)
    
    humX = humXs[frame]
    humY = humYs[frame]

    humZ = humThets[frame]
    humTrajX.append(humX)
    humTrajY.append(humY)
    humTrajZ.append(humZ)
    reldist = np.sqrt((robX - humX)**2 + (robY - humY)**2)
    if reldist < closest:
      closest = reldist
      closestT = frame
    t += 0.5
    frame += 1
  #print('dist traveled')
  #print(dist)
  #print('start frame')
  #print(initFrame)
#
  #print('closest frame')
  #print(closestT )
  #print('closest dist')
  #print(closest)
#
  print('Robot Stopping Trajectory')
  print("Robot X: ", robTrajX)
  print("Robot y: ", robTrajY)
  print(np.ones_like(robTrajY)*robThet)
  print('Human trajectory')
  print("Hum X: ", humTrajX)
  print("Hum y: ", humTrajY)
  print("Hum heading: ", humTrajZ)
  return closest

def stopAccelTime(egoVel, stopDist):
  # can't stop 
  accel = -egoVel**2/(2*stopDist)
  maxDecel = -10
  # only deccelerate as much as physically possible
  if accel < maxDecel:
    accel = maxDecel
  stopTime = -egoVel/accel

  return accel, stopTime
if __name__ == "__main__":
  print(main('uturn'))
  print(main('stop'))

