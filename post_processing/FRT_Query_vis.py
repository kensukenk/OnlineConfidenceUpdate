import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('../Trajectron-plus-plus/trajectron')
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import numpy as np
import torch
import dill
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from helper import *
from numpy.linalg import lstsq
from scipy.stats import multivariate_normal, norm
# Enable import from parent package

import matplotlib.pyplot as plt

import math

from visualization_helper import visGraph, queryFRT

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

# filters control inputs to make sure min < max and within actuatable limits
def filter_ctrl(ctrls):
  omin1, omax1, amin1, amax1, omin2, omax2, amin2, amax2 = ctrls
  os = [omin1, omax1, omin2,  omax2]
  accs = [amin1, amax1, amin2, amax2]
  if os[0] > os[1]:
    os[0], os[1] = os[1], os[0]
  if os[2] > os[3]:
    os[2], os[3] = os[3], os[2]
  if accs[0] > accs[1]:
    accs[0], accs[1] = accs[1], accs[0]
  if accs[2] > accs[3]:
    accs[2], accs[3] = accs[3], accs[2]

  for idx in range(len(os)):
    if os[idx] < -2/3:
      os[idx] = -2/3
    if os[idx] > 2/3:
      os[idx] = 2/3
    
  for idx in range(len(accs)):
    if accs[idx] < -10:
      accs[idx] = -10
    if accs[idx] > 10:
      accs[idx] = 10

  return os[0], os[1], accs[0], accs[1], os[2], os[3], accs[2], accs[3]

def get_ctrls(mode, beta_low, gamma):
  nuScenes_data_path = './v1.0' # Data Path to nuScenes data set 
  nuScenes_devkit_path = './devkit/python-sdk/'
  sys.path.append(nuScenes_devkit_path)

  # NuScenes parameters relevant to the scene
  if mode == 'uturn':
    with open('../Trajectron-plus-plus/experiments/processed/nuScenes_uturntest_full.pkl', 'rb') as f:
      print('U TURN')
      eval_env = dill.load(f, encoding='latin1')
      eval_scenes = eval_env.scenes
      hum_idx = 1
      rob_idx = 0
      scene_idx = 1
      frames = torch.arange(4,26, 1)

  elif mode == 'stop':
    with open('../Trajectron-plus-plus/experiments/processed/nuScenes_ICRAtest_full.pkl', 'rb') as f:
      print('STOP SIGN')
      eval_env = dill.load(f, encoding='latin1')
      eval_scenes = eval_env.scenes
      hum_idx = 1
      rob_idx = -1
      scene_idx = 0
      frames = torch.arange(1,19, 1)

  else:
    raise Exception("Only U-turn or Stop sign scenarios are allowed.")

  # probability mass bounds
  prob_min = 0.5 - gamma/2
  prob_max = 0.5 + gamma/2

  ph = 6
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
  mus = []
  covs = []
  covsBel = []

  humVel =  np.zeros(len(frames))
  humX =    np.zeros(len(frames))
  humY =    np.zeros(len(frames))
  humThet = np.zeros(len(frames))
  robVel =  np.zeros(len(frames))
  robX =    np.zeros(len(frames))
  robY =    np.zeros(len(frames))
  robThet = np.zeros(len(frames))


  beliefs = [0.5] # Initial prior
  beta_interp = [(beta_low + 1)/2]
  eps = 1e-8 # amount to sample from initial prior (see: Confidence-aware motion prediction for real-time collision avoidance, Fridovich-Keil)

  pred_states = np.zeros([6,2,len(frames)])
  t_idx = 0
  # storing actions, beliefs, and prediction mean and covariances
  with torch.no_grad():
      for i in range(len(frames)):
          tstep = int(frames[i]) + 1
          timestep = np.array([tstep])
          
          predictions_mm_actions, predictions_mm_sig = eval_stg.predict_actions(scene,
                                            timestep,
                                            ph,
                                            num_samples=1,
                                            z_mode=True,
                                            gmm_mode=True)
          
          predictions_mm = eval_stg.predict(scene,
                                          timestep,
                                          ph,
                                          num_samples=1,
                                          z_mode=True,
                                          gmm_mode=True)
          
         
          keys = list(predictions_mm[tstep].keys())
          pred_states[:,:,t_idx] = predictions_mm[tstep][keys[1]][0,0,:,:]
          t_idx += 1
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
          humX[i] = scene.nodes[hum_idx].data.data[tstep][0]
          humY[i] = scene.nodes[hum_idx].data.data[tstep][1]
          humVel[i] = velHum
          
          # combatting noise in sensor measurements
          if mode == 'uturn':
            if i < 4 and i > 0:
              print('combat noise')
              humThet[i] = humThet[0]
            else:
              humThet[i] = vel_ang
          else:
            humThet[i] = vel_ang
          
          vxRob = scene.nodes[rob_idx].data.data[tstep][2]
          vyRob = scene.nodes[rob_idx].data.data[tstep][3]
          vel_angRob = np.arctan2(vyRob,vxRob)
          velRob = np.sqrt(vxRob*vxRob + vyRob*vyRob)
          robX[i] = scene.nodes[rob_idx].data.data[tstep][0]
          robY[i] = scene.nodes[rob_idx].data.data[tstep][1]
          robVel[i] = velRob
          robThet[i] = vel_angRob
     
          acc_ang = np.arctan2(ay2,ax2)
          if np.abs(vel_ang - acc_ang) > 1:
              accel = -np.sqrt(ax2*ax2 + ay2*ay2)
          else:
              accel = np.sqrt(ax2*ax2 + ay2*ay2)


          var = multivariate_normal(mean=pred_agent_act[0], cov=pred_agent_sig[0])
          prob1 = var.pdf([steering, accel]) # probability of beta high
          var2 = multivariate_normal(mean=pred_agent_act[0], cov=(1/beta_low)*pred_agent_sig[0])
          prob2 = var2.pdf([steering, accel]) # probability of beta low
          
          prior = (1-eps)*beliefs[-1] + eps*beliefs[0]
          
          #              b_(beta_high)*f() / sum( f() * b_(beta))          
          beliefs.append(prior*prob1 / (prior*prob1 + (1 - prior)*prob2))
        
          newMu = pred_agent_act
          
          beta_interp.append(beliefs[-1] + beta_low*(1-beliefs[-1]))
          newCovBel = (1/beta_interp[-1])*pred_agent_sig
        
          newCov = pred_agent_sig
          
          mus.append(newMu)
          covs.append(newCov)
          covsBel.append(newCovBel)

  print('beta interp')
  print(beta_interp)
  newMu = mus[-1]
  newCovBel = covsBel[-1]
  newCov = covs[-1]

  ctrls = np.zeros([8,len(frames)])
  ctrls_bel = np.zeros([8,len(frames)])
  
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
    ctrls[:,t] = [(b_omin)[0],(b_omax)[0], (b_amin)[0], (b_amax)[0], (b_omin + A_omin*3)[0],(b_omax + A_omax*3)[0] ,(b_amin + A_amin*3)[0], (b_amax + A_amax*3)[0]]
    ctrls_bel[:,t] = [(b_ominBel)[0],(b_omaxBel)[0], (b_aminBel)[0], (b_amaxBel)[0], (b_ominBel + A_ominBel*3)[0],(b_omaxBel + A_omaxBel*3)[0] ,(b_aminBel + A_aminBel*3)[0], (b_amaxBel + A_amaxBel*3)[0]]
  return ctrls, ctrls_bel, [humX, humY, humThet, humVel], pred_states, [robX, robY, robThet], frames, beta_interp

def main(mode, beta_low, gamma):
  ph = 6   # prediction horizon
  ctrls, ctrls_bel, humStates, pred_states, robStates, frames, beta_interp = get_ctrls(mode, beta_low, gamma)
  humX, humY, humThet, humVel = humStates
  robX, robY, robThet = robStates
  collision = [0]
  collisionBel = [0]
  # scene based starting and stopping parameters
  if mode =='uturn':
    stopX = 115.89
    stopY = 83.7
    egoVel = 21
    egoThet = 2.5
    robStartX = robX[11] - egoVel*np.cos(egoThet)*0.5*12
    robStartY = robY[11] - egoVel*np.sin(egoThet)*0.5*12+1.3
  if mode == 'stop':
    stopX = robX[7]
    stopY = robY[7]
    robStartX = robX[0] + 3.5*6.8
    robStartY = robY[0] + 5.5*6.8
    egoThet = robThet[0]
    egoVel = 21


  # check predictions and FRT
  rob_xTrajPast = []
  rob_yTrajPast = []
  rob_xTrajConfPast = []
  rob_yTrajConfPast = []
  robTrajPast = [[],[]]
  robTrajConfPast = [[],[]]
  for frame in range(len(frames)):
    # start state that relative position measured from
    humStartX = humX[frame]
    humStartY = humY[frame]
    humStartThet = humThet[frame]
    humStartVel = humVel[frame]
    humPred = pred_states[:,:,frame]
    egoStartX = robStartX + egoVel*np.cos(egoThet)*frame*0.5
    egoStartY = robStartY + egoVel*np.sin(egoThet)*frame*0.5
    egoX = robStartX + egoVel*np.cos(egoThet)*frame*0.5
    egoY = robStartY + egoVel*np.sin(egoThet)*frame*0.5    
    
    omin1, omax1, amin1, amax1, omin2, omax2, amin2, amax2  = filter_ctrl(ctrls[:,frame])
    omin1Bel, omax1Bel, amin1Bel, amax1Bel, omin2Bel, omax2Bel,  amin2Bel, amax2Bel = filter_ctrl(ctrls_bel[:,frame])
    
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

    humState = [humStartX, humStartY, humStartThet]
    robState = [egoStartX, egoStartY, egoThet]
    for t in range(ph+1):   
      egoXOld = egoX 
      egoYOld = egoY
      egoThetOld = egoThet

      egoX += egoVel*np.cos(egoThet)*0.5
      egoY += egoVel*np.sin(egoThet)*0.5
      rob_xTraj.append(rob_xTraj[-1]+egoVel*np.cos(egoThet)*0.5)         
      rob_yTraj.append(rob_yTraj[-1]+egoVel*np.sin(egoThet)*0.5)
      rob_zTraj.append(egoThet)

    egoStartX = robStartX + egoVel*np.cos(egoThet)*frame*0.5
    egoStartY = robStartY + egoVel*np.sin(egoThet)*frame*0.5
    egoX = robStartX + egoVel*np.cos(egoThet)*frame*0.5
    egoY = robStartY + egoVel*np.sin(egoThet)*frame*0.5  
    egoXOld = None
    egoYOld = None
    egoThetOld = None
    bounds1 = [amin1, amax1, omin1, omax1, amin2, amax2, omin2, omax2]
    bounds2 = [amin1Bel, amax1Bel, omin1Bel, omax1Bel, amin2Bel, amax2Bel, omin2Bel, omax2Bel]
    bounds = [bounds1, bounds2]

    humHistX = humX[:frame]
    humHistY = humY[:frame]
    humHist = [humHistX, humHistY]
    robPred = [rob_xTraj, rob_yTraj]


    if mode == 'uturn':
      if collisionBel != [0] and len(collisionBel[0]) != 0:
        colBelx = collisionBel[0][1:]
        colBely = collisionBel[1][1:]
        robBelx = collisionBel[0][0]
        robBely = collisionBel[1][0]
        robBelState = [robBelx, robBely, egoThet]
        rob_xTrajConfPast.append(robBelx)
        rob_yTrajConfPast.append(robBely)
        robTrajConfPast = [rob_xTrajConfPast, rob_yTrajConfPast]
        if collision != [0] and len(collision[0]) != 0:
          colx = collision[0][1:]
          coly = collision[1][1:]
          robNobelx = collision[0][0]
          robNobely = collision[1][0]
          robNobelState = [robNobelx, robNobely, egoThet]
          rob_xTrajPast.append(robNobelx)
          rob_yTrajPast.append(robNobely)
          robTrajPast = [rob_xTrajPast, rob_yTrajPast]
          visGraph(mode,  humStartVel, bounds, humHist, humPred, humState, [robNobelState, robBelState], [collision, collisionBel], [robTrajPast, robTrajConfPast], frame,  beta_interp)
          collision = [colx, coly]
          collisionBel = [colBelx, colBely]
        else:
          rob_xTrajPast.append(egoStartX)
          rob_yTrajPast.append(egoStartY)
          robTrajPast = [rob_xTrajPast, rob_yTrajPast]
          visGraph(mode, humStartVel, bounds, humHist, humPred, humState, [robState, robBelState], [robPred, collisionBel], [robTrajPast, robTrajConfPast], frame,  beta_interp)
          collisionBel = [colBelx, colBely]
      else:
        rob_xTrajPast.append(egoStartX)
        rob_yTrajPast.append(egoStartY)
        robTrajPast = [rob_xTrajPast, rob_yTrajPast]
        rob_xTrajConfPast.append(egoStartX)
        rob_yTrajConfPast.append(egoStartY)
        robTrajConfPast = [rob_xTrajConfPast, rob_yTrajConfPast]
        visGraph(mode, humStartVel, bounds, humHist, humPred, humState, [robState, robState], [robPred, robPred], [robTrajPast, robTrajConfPast],  frame,  beta_interp)


    for t in range(ph+1):   
      egoXOld = egoX 
      egoYOld = egoY
      egoThetOld = egoThet
      egoX += egoVel*np.cos(egoThet)*0.5
      egoY += egoVel*np.sin(egoThet)*0.5
      # want to check over range of points in the time interval
      egoXRange = np.linspace(egoXOld, egoX, 15)
      egoYRange = np.linspace(egoYOld, egoY, 15)
      egoThetRange = np.linspace(egoThetOld, egoThet, 15)
      for i in range(len(egoXRange)):
        relX, relY, _ = relativeCoords(humStartX, humStartY, humStartThet, egoXRange[i], egoYRange[i], egoThetRange[i])
        querypoint = queryFRT(relX, relY, humStartVel, amin1, amax1, omin1, omax1, amin2, amax2, omin2, omax2)
        querypoint_bel = queryFRT(relX, relY, humStartVel, amin1Bel, amax1Bel, omin1Bel, omax1Bel, amin2Bel, amax2Bel, omin2Bel, omax2Bel)
        if collisionBel == [0]:
          collisionBel = collisionCheck('Belief', querypoint_bel, frame, t, egoStartX, egoStartY, egoThet, egoVel, humX, humY, humThet, stopX, stopY, rob_xTrajStop, rob_yTrajStop, hum_xTrajStop, hum_yTrajStop, hum_zTrajStop)
      
        if collision == [0]:
          collision = collisionCheck('No Belief', querypoint, frame, t, egoStartX, egoStartY, egoThet, egoVel, humX, humY, humThet, stopX, stopY, rob_xTrajStop, rob_yTrajStop, hum_xTrajStop, hum_yTrajStop, hum_zTrajStop)              
  

def collisionCheck(belief_mode, queryVal, frame, t, egoStartX, egoStartY, egoThet, egoVel, humX, humY, humThet, stopX, stopY,rob_xTrajStop, rob_yTrajStop, hum_xTrajStop, hum_yTrajStop, hum_zTrajStop):
  if queryVal < 0:
    stopDist = np.sqrt((egoStartX-stopX)**2 + (egoStartY-stopY)**2)
    if stopX > egoStartX:
      stopDist = -stopDist
    print(belief_mode +":")
    print('distance from stop')
    print(stopDist)
    closest_Dist, robTrajX, robTrajY = closestDist(frame, stopDist, egoStartX, egoStartY, egoThet, egoVel, humX, humY, humThet, rob_xTrajStop, rob_yTrajStop, hum_xTrajStop, hum_yTrajStop, hum_zTrajStop)
    print('Closest Dist')
    print(closest_Dist)
    return [robTrajX, robTrajY]
  else:
    return [0]

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

  #print('Robot Stopping Trajectory')
  #print("Robot X: ", robTrajX)
  #print("Robot y: ", robTrajY)
  #print(np.ones_like(robTrajY)*robThet)
  #print('Human trajectory')
  #print("Hum X: ", humTrajX)
  #print("Hum y: ", humTrajY)
  #print("Hum heading: ", humTrajZ)

  


  return closest, robTrajX, robTrajY

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

  #beta_lows = [0.01, 0.03, 0.05, 0.07]
  #beta_lows = [0.02, 0.04, 0.06, 0.08]
  beta_lows = [0.03]
  gammas = [0.025, 0.05, 0.075, 0.1, 0.125]
  gammas = [0.075]
  for gamma in gammas:
    #print(main('uturn', 0.03, 0.075)) # 4.5
    print(main('stop', 0.03, 0.075)) #2.0

