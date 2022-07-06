import csv
import glob
import math
import os

import matplotlib.colors as colors
import numpy as np
import scipy.io as spio
import torch
from torch.utils.data import Dataset
import diff_operators
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

import utils
import pickle

from scipy.stats import truncnorm


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()


def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()

def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)


class ReachabilityMultiVehicleCollisionSourceNE(Dataset):
    def __init__(self, numpoints,
     collisionR=0.25, velocity=0.6, omega_max=1.1,
     pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3, 
     numEvaders=1, pretrain_iters=2000, angle_alpha=1.0, time_alpha=1.0,
     num_src_samples=1000):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.velocity = velocity
        self.omega_max = omega_max
        self.collisionR = collisionR

        self.alpha_angle = angle_alpha * math.pi
        self.alpha_time = time_alpha

        self.numEvaders = numEvaders
        self.num_states_per_vehicle = 3
        self.num_states = self.num_states_per_vehicle * (numEvaders + 1)
        self.num_pos_states = 2 * (numEvaders + 1)
        # The state sequence will be as follows
        # [x-y position of vehicle 1, x-y position of vehicle 2, ...., x-y position of vehicle N, heading of vehicle 1, heading of vehicle 2, ...., heading of vehicle N]

        self.tMin = tMin
        self.tMax = tMax

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            # time = torch.zeros(self.numpoints, 1).uniform_(start_time - 0.001, start_time + 0.001)
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = tMin and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
        
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # set up the initial value function
        # Collision cost between the pursuer and the evaders
        boundary_values = torch.norm(coords[:, 1:3] - coords[:, 3:5], dim=1, keepdim=True) - self.collisionR
        for i in range(1, self.numEvaders):
            boundary_values_current = torch.norm(coords[:, 1:3] - coords[:, 2*(i+1)+1:2*(i+1)+3], dim=1, keepdim=True) - self.collisionR
            boundary_values = torch.min(boundary_values, boundary_values_current)
        # Collision cost between the evaders themselves
        for i in range(self.numEvaders):
            for j in range(i+1, self.numEvaders):
                evader1_coords_index = 1 + (i+1)*2
                evader2_coords_index = 1 + (j+1)*2
                boundary_values_current = torch.norm(coords[:, evader1_coords_index:evader1_coords_index+2] - coords[:, evader2_coords_index:evader2_coords_index+2], dim=1, keepdim=True) - self.collisionR
                boundary_values = torch.min(boundary_values, boundary_values_current)

        # normalize the value function
        norm_to = 0.02
        mean = 0.25
        var = 0.5
        boundary_values = (boundary_values - mean)*norm_to/var
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}


class ReachabilityAir3DSource(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.25, velocity=0.6, omega_max=1.1, 
        pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, angle_alpha=1.0, num_src_samples=1000, seed=0):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.velocity = velocity
        self.omega_max = omega_max
        self.collisionR = collisionR

        self.alpha_angle = angle_alpha * math.pi

        self.num_states = 3

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # set up the initial value function
        boundary_values = torch.norm(coords[:, 1:3], dim=1, keepdim=True) - self.collisionR

        # normalize the value function
        norm_to = 0.02
        mean = 0.25
        var = 0.5

        boundary_values = (boundary_values - mean)*norm_to/var
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}

class ReachabilityHumanForward(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.1, velocity=1.0, omega_max=1.1, 
        pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, angle_alpha=1.0, num_src_samples=1000, beta1 = 0.1, beta2 = 10, seed=0):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.velocity = velocity
        self.collisionR = collisionR

        self.alpha_angle = angle_alpha * math.pi

        self.num_states = 5 #states are x,y, startx, starty, p

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.goal = torch.tensor([0.0, 0.0])
        self.beta1 = beta1
        self.beta2 = beta2
        
        

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # set up the initial value function
        boundary_values = torch.norm(coords[:, 1:3]-coords[:,3:5], dim=1, keepdim=True) - self.collisionR

        # normalize the value function
        norm_to = 0.02
        mean = 0.25
        var = 0.5

        boundary_values = (boundary_values - mean)*norm_to/var
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}


class HumanRobotPE(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.1, velocity=1.0, omega_max=1.1, 
        pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, angle_alpha=1.0, num_src_samples=1000, beta1 = 0.1, beta2 = 10, periodic_boundary = False, seed=0):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.periodic_boundary = periodic_boundary
        self.numpoints = numpoints
        
        self.velocity = velocity
        self.collisionR = collisionR

        self.alpha_angle = 1.2 * math.pi
        self.omega_max = omega_max

        self.num_states = 6 #states are x_r, y_r, theta_r, x_h, y_h, p
        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples
        self.N_boundary_pts = self.N_src_samples//2

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.goal = torch.tensor([0.0, 0.0])
        self.beta1 = beta1
        self.beta2 = beta2
        
        

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions
        angle_index = 3 #index of angle state

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # Sample some points to impose the boundary coditions
        if self.periodic_boundary:
            # import ipdb; ipdb.set_trace()
            coords_angle = torch.zeros(self.N_boundary_pts, 1).uniform_(math.pi-0.001, self.alpha_angle) # Sample near the right boundary
            coords_angle[0:self.N_boundary_pts//2] = -1.0 * coords_angle[0:self.N_boundary_pts//2] # Assign half of the points to the left boundary
            coords_angle_periodic = angle_normalize(coords_angle)
            coords_angle_concatenated = torch.cat((coords_angle, coords_angle_periodic), dim=0)
            coords_angle_concatenated_normalized = (coords_angle_concatenated)/self.alpha_angle
            coords[:self.N_boundary_pts] = coords[self.N_boundary_pts:2*self.N_boundary_pts]
            coords[:2*self.N_boundary_pts, angle_index] = coords_angle_concatenated_normalized[..., 0]    

        # set up the initial value function
        boundary_values = torch.norm(coords[:, 1:3]-coords[:,4:6], dim=1, keepdim=True) - self.collisionR



        # normalize the value function
        norm_to = 0.02
        mean = 0.25
        var = 0.5

        boundary_values = (boundary_values - mean)*norm_to/var
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}


class ReachabilityHumanForwardParam(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.1, velocity=1.0, omega_max=1.1, 
        pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, angle_alpha=1.0, num_src_samples=1000, periodic_boundary=False, diffModel=False, seed=0):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.periodic_boundary = periodic_boundary
        self.diffModel = diffModel
        self.numpoints = numpoints
        
        self.velocity = velocity
        self.collisionR = collisionR

        self.alpha_angle = angle_alpha * math.pi
        # t, x,y, x0,y0, umin1, umax1, umin2, umax2, umin3, umax3, umin4, umax4, umin5, umax5

        self.num_states = 6 #states are x,y, startx, starty, umin1, umax1
        self.num_angle_param = 2
        self.umin_index = 5
        self.umax_index = 6

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples
        self.N_boundary_pts = self.N_src_samples//2

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.norm_to = 0.02
        self.mean = 0.25
        self.var = 0.5

        
        #self.goal = torch.tensor([0.0, 0.0])
        #self.beta1 = beta1
        #self.beta2 = beta2
        
        

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1
    
    def compute_IC(self, state_coords):
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[..., 0:2] = state_coords_unnormalized[..., 0:2] - state_coords_unnormalized[..., 2:4]
        #state_coords_unnormalized[..., 1] = state_coords_unnormalized[..., 1] - state_coords_unnormalized[..., 3]
        #state_coords_unnormalized[..., 2] = state_coords_unnormalized[..., 2] * self.alpha['th'] + self.beta['th']
        #boundary_values = torch.norm(state_coords[:, 1:3]-state_coords[:,3:5], dim=1, keepdim=True) - self.collisionR
        boundary_values = torch.norm(state_coords_unnormalized[..., 0:2], dim=-1, keepdim=True) - self.collisionR
        return boundary_values

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions


        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        if self.periodic_boundary:
            coords_angle = torch.zeros(self.N_boundary_pts, self.num_angle_param).uniform_(math.pi-0.001, self.alpha_angle)
            coords_angle[0:self.N_boundary_pts//2] = -1.0 * coords_angle[0:self.N_boundary_pts//2] # Assign half of the points to the left boundary
            coords_angle_periodic = angle_normalize(coords_angle)
            coords_angle_concatenated = torch.cat((coords_angle, coords_angle_periodic), dim=0)
            coords_angle_concatenated_normalized = (coords_angle_concatenated )/self.alpha_angle
            coords[:self.N_boundary_pts] = coords[self.N_boundary_pts:2*self.N_boundary_pts]
            coords[:2*self.N_boundary_pts, self.umin_index:self.umax_index+1] = coords_angle_concatenated_normalized[..., :]



        # set up the initial value function
        # Compute the initial value function
        if self.diffModel:
            coords_var = torch.tensor(coords.clone(), requires_grad=True)
            boundary_values = self.compute_IC(coords_var[..., 1:])
            #boundary_values = torch.norm(coords_var[:, 1:3]-coords_var[:,3:5], dim=1, keepdim=True) - self.collisionR

            # normalize the value function
            #norm_to = 0.02
            #mean = 0.25
            #var = 0.5

            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            # Compute the gradients of the value function
            lx_grads = diff_operators.gradient(boundary_values, coords_var)[..., 1:]
        else:
            boundary_values = self.compute_IC(coords[..., 1:])

            #boundary_values = torch.norm(coords[:, 1:3]-coords[:,3:5], dim=1, keepdim=True) - self.collisionR

            # normalize the value function
            #norm_to = 0.02
            #mean = 0.25
            #var = 0.5

            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            #boundary_values = self.compute_IC(coords[..., 1:])

            # Normalize the value function
            # print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            #boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            # print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        


        #boundary_values = torch.norm(coords[:, 1:3]-coords[:,3:5], dim=1, keepdim=True) - self.collisionR

        # normalize the value function
        #norm_to = 0.02
        #mean = 0.25
        #var = 0.5

        #boundary_values = (boundary_values - mean)*norm_to/var
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.diffModel:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx_grads': lx_grads}
        else:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}