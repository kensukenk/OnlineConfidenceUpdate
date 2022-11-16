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
    elif dim == 4:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2],  :sidelen[3]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[0] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[3] - 1)
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

        self.goal = torch.tensor([1.0, -1.0])
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
        self.num_states = 14 #states are x,y, startx, starty, umin1, umax1, umin2, umax2, umin3, umax3, umin4, umax4, umin5, umax5
        

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

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1
    
    def compute_IC(self, state_coords):
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[..., 0:2] = state_coords_unnormalized[..., 0:2] - state_coords_unnormalized[..., 2:4]
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

        
        # set up the initial value function
        # Compute the initial value function
        if self.diffModel:
            coords_var = torch.tensor(coords.clone(), requires_grad=True)
            boundary_values = self.compute_IC(coords_var[..., 1:])

            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            # Compute the gradients of the value function
            lx_grads = diff_operators.gradient(boundary_values, coords_var)[..., 1:]
        else:
            boundary_values = self.compute_IC(coords[..., 1:])
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var

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

class ReachabilityDubins4DForwardParam6set(Dataset):
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
        

        self.num_states = 30 #states are x,y, theta, v, startx, starty, amin1, amax1, omegamin1, omegamax1, amin2, amax2, omegamin2, omegamax2
        self.angle_index = 3

        self.tMax = tMax
        self.tMin = tMin

        self.collisionR = collisionR

        # Define state alphas and betas so that all coordinates are from [-1, 1]. 
        # The conversion rule is state' = (state - beta)/alpha. state' is in [-1, 1].
        self.alpha = {}
        self.beta = {}
        self.alpha['x'] = 10.0
        self.alpha['y'] = 10.0
        self.alpha['th'] = 1.2*math.pi
        self.alpha['v'] = 30.0
        self.alpha['a'] = 20.0
        self.alpha['o'] = 3.*math.pi
        # self.alpha['time'] = 2.0/self.tMax
        #self.alpha['time'] = 3.0/self.tMax
        self.alpha['time'] = 1.0

        self.beta['x'] = 0.0
        self.beta['y'] = 0.0
        self.beta['th'] = 0.0
        self.beta['v'] = 25.0
        self.beta['a'] = 0.0
        self.beta['o'] = 0.0

        # Scale for output normalization
        self.norm_to = 0.02
        self.mean = 0.25 
        self.var = 0.5

        # State bounds
        self.vMin = 0.001
        self.vMax = 6.50
        #self.thMin = -0.3*math.pi + 0.001
        #self.thMax = 0.3*math.pi - 0.001

        self.N_src_samples = num_src_samples
        self.N_boundary_pts = self.N_src_samples//2

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        
        # Set the seed
        torch.manual_seed(seed)

    def time_control(self, time, set1, set2, set3, set4, set5, set6):      
        
        time_control = (1 - 5*time)*set1 + (5*time)*set2
        time_control = torch.where(time > 0.2, (1 - 5*(time- 0.2))*set2 + (5*(time-0.2))*set3, time_control)  
        time_control = torch.where(time > 0.4, (1 - 5*(time- 0.4))*set3 + (5*(time-0.4))*set4, time_control)
        time_control = torch.where(time > 0.6, (1 - 5*(time- 0.6))*set4 + (5*(time-0.6))*set5, time_control) 
        time_control = torch.where(time > 0.8, (1 - 5*(time- 0.8))*set5 + (5*(time-0.8))*set6, time_control) 
        time_control = torch.where(time > 1.0, set6, time_control)


        return time_control
    def compute_overall_ham(self, x, dudx, return_components=False):
        alpha = self.alpha
        beta = self.beta

        # Scale the costates appropriately.
        dudx[..., 0] = dudx[..., 0] / alpha['x']
        dudx[..., 1] = dudx[..., 1] / alpha['y']
        dudx[..., 2] = dudx[..., 2] / alpha['th']
        dudx[..., 3] = dudx[..., 3] / alpha['v']

        # Scale the states appropriately.
        x_u = x * 1.0
        #x_u[..., 0] = x_u[..., 0] * alpha['time']
        x_u[..., 1] = x_u[..., 1] * alpha['x'] + beta['x']
        x_u[..., 2] = x_u[..., 2] * alpha['y'] + beta['y']
        x_u[..., 3] = x_u[..., 3] * alpha['th'] + beta['th']
        x_u[..., 4] = x_u[..., 4] * alpha['v'] + beta['v']
        x_u[..., 5] = x_u[..., 5] * alpha['x'] + beta['x']
        x_u[..., 6] = x_u[..., 6] * alpha['y'] + beta['y']
        x_u[..., 7] = x_u[..., 7] * alpha['a'] + beta['a'] #amin
        x_u[..., 8] = x_u[..., 8] * alpha['a'] + beta['a'] #amax
        x_u[..., 9] = x_u[..., 9] * alpha['o'] + beta['o'] # omin
        x_u[..., 10] = x_u[..., 10] * alpha['o'] + beta['o'] # omax
        x_u[..., 11] = x_u[..., 11] * alpha['a'] + beta['a'] #set 2
        x_u[..., 12] = x_u[..., 12] * alpha['a'] + beta['a']
        x_u[..., 13] = x_u[..., 13] * alpha['o'] + beta['o']
        x_u[..., 14] = x_u[..., 14] * alpha['o'] + beta['o']
        x_u[..., 15] = x_u[..., 15] * alpha['a'] + beta['a'] #set3
        x_u[..., 16] = x_u[..., 16] * alpha['a'] + beta['a']
        x_u[..., 17] = x_u[..., 17] * alpha['o'] + beta['o']
        x_u[..., 18] = x_u[..., 18] * alpha['o'] + beta['o']
        x_u[..., 19] = x_u[..., 19] * alpha['a'] + beta['a'] # set 4
        x_u[..., 20] = x_u[..., 20] * alpha['a'] + beta['a']
        x_u[..., 21] = x_u[..., 21] * alpha['o'] + beta['o']
        x_u[..., 22] = x_u[..., 23] * alpha['o'] + beta['o']
        x_u[..., 23] = x_u[..., 23] * alpha['a'] + beta['a'] #set5
        x_u[..., 24] = x_u[..., 24] * alpha['a'] + beta['a']
        x_u[..., 25] = x_u[..., 25] * alpha['o'] + beta['o']
        x_u[..., 26] = x_u[..., 26] * alpha['o'] + beta['o']
        x_u[..., 27] = x_u[..., 27] * alpha['a'] + beta['a'] #set 6
        x_u[..., 28] = x_u[..., 28] * alpha['a'] + beta['a']
        x_u[..., 29] = x_u[..., 29] * alpha['o'] + beta['o']
        x_u[..., 30] = x_u[..., 30] * alpha['o'] + beta['o']


        #amin = self.time_control(x_u[...,0], x_u[...,7],x_u[...,11],x_u[...,15],x_u[...,19],x_u[...,23],x_u[...,27])
        #amax = self.time_control(x_u[...,0], x_u[...,8],x_u[...,12],x_u[...,16],x_u[...,20],x_u[...,24],x_u[...,28])
        #omin = -self.time_control(x_u[...,0], x_u[...,9],x_u[...,13],x_u[...,17],x_u[...,21],x_u[...,25],x_u[...,29])
        #omax = -self.time_control(x_u[...,0], x_u[...,10],x_u[...,14],x_u[...,18],x_u[...,22],x_u[...,26],x_u[...,30])
        amin = (x_u[..., 7] + x_u[...,11] + x_u[...,15] + x_u[...,19] + x_u[...,23] + x_u[...,27]) / 6
        amax = (x_u[..., 8] + x_u[...,12] + x_u[...,16] + x_u[...,20] + x_u[...,24] + x_u[...,28]) / 6 
        omin = -(x_u[..., 9] + x_u[...,13] + x_u[...,17] + x_u[...,21] + x_u[...,25] + x_u[...,29]) / 6
        omax = -(x_u[...,10] + x_u[...,14] + x_u[...,18] + x_u[...,22] + x_u[...,26] + x_u[...,30]) / 6       
        

        zero_tensor = torch.Tensor([0]).cuda()
        #amin = torch.where((x_u[..., 4] <= self.vMin), zero_tensor, amin)
        #amax = torch.where((x_u[..., 4] >= self.vMax), zero_tensor, amax)
        
        o_opt = torch.where(dudx[...,2]>0, omin, omax)
        a_opt = torch.where(dudx[...,3]>0, amax, amin)
        # xdot = v cos theta
        # ydot = v sin theta
        # thetadot = o_opt
        # vdot = a_opt
        # negative dynamics since it is a FRS, want to minimize
        ham = -dudx[...,0]*x_u[...,4]*(torch.cos(x_u[..., 3])) - dudx[...,1]*x_u[...,4]*(torch.sin(x_u[...,3]))
        ham_o = -dudx[...,2]*o_opt
        ham_a = -dudx[...,3]*a_opt
        ham = ham + ham_a + ham_o       
        
        return ham

    def __len__(self):
        return 1
    
    def compute_lx(self, state_coords_unnormalized):
        # Compute the target boundary condition given the unnormalized state coordinates.
        # Vehicle 1
        goal_tensor_R1 = state_coords_unnormalized[:, 4:6]
        dist_R1 = torch.norm(state_coords_unnormalized[:, 0:2] - goal_tensor_R1, dim=1, keepdim=True) - self.collisionR
        return dist_R1
    
    def compute_IC(self, state_coords):
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[:, 0] = state_coords_unnormalized[:, 0] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 1] = state_coords_unnormalized[:, 1] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[:, 4] = state_coords_unnormalized[:, 4] * self.alpha['x'] + self.beta['x'] # start x
        state_coords_unnormalized[:, 5] = state_coords_unnormalized[:, 5] * self.alpha['y'] + self.beta['y'] # start y

        lx = self.compute_lx(state_coords_unnormalized)
        return lx

    
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
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # Sample some points to impose the boundary coditions
        if self.periodic_boundary:
            # import ipdb; ipdb.set_trace()
            coords_angle = torch.zeros(self.N_boundary_pts, 1).uniform_(math.pi-0.001, self.alpha['th'] + self.beta['th']) # Sample near the right boundary
            coords_angle[0:self.N_boundary_pts//2] = -1.0 * coords_angle[0:self.N_boundary_pts//2] # Assign half of the points to the left boundary
            coords_angle_periodic = angle_normalize(coords_angle)
            coords_angle_concatenated = torch.cat((coords_angle, coords_angle_periodic), dim=0)
            coords_angle_concatenated_normalized = (coords_angle_concatenated - self.beta['th'])/self.alpha['th']
            coords[:self.N_boundary_pts] = coords[self.N_boundary_pts:2*self.N_boundary_pts]
            coords[:2*self.N_boundary_pts, self.angle_index] = coords_angle_concatenated_normalized[..., 0]  

        coords[..., 6:] = coords[..., 6:] * 0.0 # set controls to be constant

        # Compute the initial value function
        if self.diffModel:
            coords_var = torch.tensor(coords.clone(), requires_grad=True)
            boundary_values = self.compute_IC(coords_var[:, 1:])
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            lx_grads = diff_operators.gradient(boundary_values, coords_var)[..., 1:5] # only care about gradient wrt state inputs

        else:
            # lx, gx, boundary_values = self.compute_IC(coords[:, 1:])
            boundary_values = self.compute_IC(coords[:, 1:])
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var


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
        
class ReachabilityDubins4DForwardParam2SetScaled(Dataset):
    def __init__(self, numpoints, collisionR, tMin=0.0, tMax=1.0, alphaT=1.0,
        pretrain=False, counter_start=0, counter_end=100e3, pretrain_iters=2000, 
        num_src_samples=1000, periodic_boundary=False, diffModel=False):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.periodic_boundary = periodic_boundary
        self.diffModel = diffModel
        self.sample_inside_target = True
        self.numpoints = numpoints
        
        # Dynamics parameters
        self.num_states = 13

        # TIme parameters
        self.tMax = tMax
        self.tMin = tMin

        # Normalization for states and time
        self.alpha = {}
        self.beta = {}

        self.alpha['x'] = 4.0
        self.alpha['y'] = 4.0
        self.alpha['th'] = 1.1*math.pi
        self.alpha['v'] = 5.5
        self.alpha['a'] = 10.0
        self.alpha['o'] = 3.0
        self.alpha['time'] = 3.0

        self.beta['x'] = 3.0
        self.beta['y'] = 0.0
        self.beta['th'] = 0.0
        self.beta['v'] = 5.5
        self.beta['a'] = 0.0
        self.beta['o'] = 0.0

        # Normalization for the value function
        self.norm_to = 0.02
        self.mean = 5.2
        self.var = 5.6

        # Collision radius
        self.collisionR = collisionR
        self.vhR = 0.75

        self.N_src_samples = num_src_samples
        self.N_boundary_pts = self.N_src_samples//2
        self.N_inside_target_samples = num_src_samples


        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end

    def __len__(self):
        return 1

    def time_control(self, time, set1, set2):
        time_control = (1.0 - time)*set1 + (time)*set2
        #time_control = torch.where(time > 1.0, set2, time_control)
        return time_control

    def sample_inside_target_set(self):
        # Sample coordinates that are inside the target set.
        target_coords = torch.zeros(self.N_inside_target_samples, 5).uniform_(-1, 1)
        
        # XY position
        normalized_x_extent = (1.5*2*self.collisionR)/ self.alpha['x']
        normalized_y_extent = (1.5*2*self.collisionR)/ self.alpha['y']
        normalized_x_shift = -self.beta['x']/ self.alpha['x']
        normalized_y_shift = -self.beta['y']/ self.alpha['y']
        # sets the range from [-1, 1] to [-extent, +extent], then centers around origin
        target_coords[..., 0] = normalized_x_extent * target_coords[:, 0] + normalized_x_shift
        target_coords[..., 1] = normalized_y_extent * target_coords[:, 1] + normalized_y_shift

        # Theta position
        normalized_th_extent = (1.5* 2.0 *self.vhR - self.beta['th'])/ self.alpha['th']
        target_coords[..., 2] = normalized_th_extent * target_coords[:, 2]

        # V position
        unnormalizedV = target_coords[..., 4] * self.alpha['v'] + self.beta['v'] # Unnormalized starting speed
        speed_deviation = target_coords[..., 3] * 1.5 * 2.0 * self.vhR # Speed deviation around the starting speed
        target_coords[..., 3] = unnormalizedV + speed_deviation # Starting V around the target set
        target_coords[..., 3] = (target_coords[..., 3] - self.beta['v'])/ self.alpha['v'] # Normalize coordinates
        target_coords[..., 3] = torch.clip(target_coords[..., 3], -1.0, 1.0) # Clip to the valid grid range


        return target_coords

    def compute_IC(self, state_coords):
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[..., 0] = state_coords_unnormalized[..., 0] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[..., 1] = state_coords_unnormalized[..., 1] * self.alpha['y'] + self.beta['y']
        
        state_coords_unnormalized[..., 2] = state_coords_unnormalized[..., 2] * self.alpha['th'] + self.beta['th']
        state_coords_unnormalized[..., 3] = state_coords_unnormalized[..., 3] * self.alpha['v'] + self.beta['v']

        state_coords_unnormalized[..., 4] = state_coords_unnormalized[..., 4] * self.alpha['v'] + self.beta['v']

        # positional states from start
        boundary_values1 = torch.norm(state_coords_unnormalized[..., 0:2], dim=-1, keepdim=True) - self.collisionR
        # velocity from start
        state_coords_unnormalized_thv = state_coords_unnormalized[..., 2:4] * 1.0
        state_coords_unnormalized_thv[..., 1] = state_coords_unnormalized_thv[..., 1] - state_coords_unnormalized[..., 4]

        boundary_values2 = torch.norm(state_coords_unnormalized_thv, dim=-1, keepdim=True) - self.vhR

        #boundary_values1 = torch.norm(state_coords_unnormalized[..., 0:2] - state_coords_unnormalized[..., 4:6], dim=-1, keepdim=True) - self.collisionR
        # theta and vel from start
        #boundary_values2 = torch.norm(state_coords_unnormalized[..., 2:4] - state_coords_unnormalized[..., 6:8], dim=-1, keepdim=True) - 0.2*self.collisionR
        boundary_values = torch.max(boundary_values1, boundary_values2)

        return boundary_values

    def compute_overall_ham(self, x, dudx):
        alpha = self.alpha
        beta = self.beta

        # Scale the costates appropriately.
        dudx[..., 0] = dudx[..., 0] / alpha['x']
        dudx[..., 1] = dudx[..., 1] / alpha['y']
        dudx[..., 2] = dudx[..., 2] / alpha['th']
        dudx[..., 3] = dudx[..., 3] / alpha['v']
        
        # Scale the states appropriately.
        x_u = x * 1.0
        x_u[..., 1] = x_u[..., 1] * alpha['x'] + beta['x']
        x_u[..., 2] = x_u[..., 2] * alpha['y'] + beta['y']
        x_u[..., 3] = x_u[..., 3] * alpha['th'] + beta['th']
        x_u[..., 4] = x_u[..., 4] * alpha['v'] + beta['v']
        
        x_u[..., 5] = x_u[..., 4] * alpha['v'] + beta['v'] # start v
        
        x_u[..., 6] = x_u[..., 6] * alpha['a'] + beta['a']
        x_u[..., 7] = x_u[..., 7] * alpha['a'] + beta['a']
        x_u[..., 8] = x_u[..., 8] * alpha['o'] + beta['o']
        x_u[..., 9] = x_u[..., 9] * alpha['o'] + beta['o']
        x_u[..., 10] = x_u[..., 10] * alpha['a'] + beta['a']
        x_u[..., 11] = x_u[..., 11] * alpha['a'] + beta['a']
        x_u[..., 12] = x_u[..., 12] * alpha['o'] + beta['o']
        x_u[..., 13] = x_u[..., 13] * alpha['o'] + beta['o']
        
        amin = self.time_control(x_u[...,0], x_u[...,6], x_u[...,10])
        amax = self.time_control(x_u[...,0], x_u[...,7], x_u[...,11])
        omin = self.time_control(x_u[...,0], x_u[...,8], x_u[...,12])
        omax = self.time_control(x_u[...,0], x_u[...,9], x_u[...,13])
        
        # Negative dynamics since it is a FRS; controls want to minimize
        # xdot = -v cos theta
        # ydot = -v sin theta
        # thetadot = -o
        # vdot = -a

        # Optimal control
        o_opt = torch.where(-dudx[...,2] > 0, omin, omax)
        a_opt = torch.where(-dudx[...,3] > 0, amin, amax)
        
        # Hamiltonian
        ham = -dudx[...,0]*x_u[...,4]*(torch.cos(x_u[..., 3])) - dudx[...,1]*x_u[...,4]*(torch.sin(x_u[...,3]))
        ham_o = -dudx[...,2]*o_opt
        ham_a = -dudx[...,3]*a_opt
        ham = ham + ham_a + ham_o
        return ham

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions
        angle_index = 3 # Index of the angle state

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        # Make sure that the oMax and aMax are greater than oMin and aMin respectively. 
        # Shifting the sampling to [aMin, 1] and [oMin, 1] for aMax and oMax respectively
        coords[..., 6] =  (0.5 * (1 - coords[..., 5]) * coords[..., 6]) + (0.5 * (1 + coords[..., 5]))
        coords[..., 10] = (0.5 * (1 - coords[..., 9]) * coords[..., 10]) + (0.5 * (1 + coords[..., 9]))
        coords[..., 8] = (0.5 * (1 - coords[..., 7]) * coords[..., 8]) + (0.5 * (1 + coords[..., 7])) 
        coords[..., 12] = (0.5 * (1 - coords[..., 11]) * coords[..., 12]) + (0.5 * (1 + coords[..., 11]))

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
            coords_angle = torch.zeros(self.N_boundary_pts, 1).uniform_(math.pi-0.001, self.alpha['th'] + self.beta['th']) # Sample near the right boundary
            coords_angle[0:self.N_boundary_pts//2] = -1.0 * coords_angle[0:self.N_boundary_pts//2] # Assign half of the points to the left boundary
            coords_angle_periodic = angle_normalize(coords_angle)
            coords_angle_concatenated = torch.cat((coords_angle, coords_angle_periodic), dim=0)
            coords_angle_concatenated_normalized = (coords_angle_concatenated - self.beta['th'])/self.alpha['th']
            coords[:self.N_boundary_pts] = coords[self.N_boundary_pts:2*self.N_boundary_pts]
            coords[:2*self.N_boundary_pts, angle_index] = coords_angle_concatenated_normalized[..., 0]

        # Add some samples that are inside the target set
        if self.sample_inside_target:
            target_coords = coords[:self.N_inside_target_samples] * 1.0 
            target_coords[..., 1:6] = self.sample_inside_target_set()
            coords = torch.cat((coords, target_coords), dim=0)

        # Compute the initial value function
        if self.diffModel:
            coords_var = torch.tensor(coords.clone(), requires_grad=True)
            boundary_values = self.compute_IC(coords_var[..., 1:])
            
            # Normalize the value function
            #print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            #print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))

            # Compute the gradients of the value function
            lx_grads = diff_operators.gradient(boundary_values, coords_var)[..., 1:5]
        else:
            boundary_values = self.compute_IC(coords[..., 1:])

            # Normalize the value function
            # print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            # print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        
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


class ReachabilityDubins4DReachAvoidParam2SetScaled(Dataset):
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

        self.num_states = 13 #states are x_rel,y_rel, theta_rel, v_r, v_h, amin1, amax1, omegamin1, omegamax1, amin2, amax2, omegamin2, omegamax2
        self.angle_index = 3

        self.tMax = tMax
        self.tMin = tMin

        # Define state alphas and betas so that all coordinates are from [-1, 1]. 
        # The conversion rule is state' = (state - beta)/alpha. state' is in [-1, 1].
        self.alpha = {}
        self.beta = {}
        self.alpha['x'] = 10.0
        self.alpha['y'] = 10.0
        self.alpha['th'] = 1.2*math.pi
        self.alpha['v'] = 7.5
        self.alpha['a'] = 10.0
        self.alpha['o'] = 3.*math.pi
        self.alpha['time'] = 3.0

        self.beta['x'] = 0.0
        self.beta['y'] = 0.0
        self.beta['th'] = 0.0
        self.beta['v'] = 7.5
        self.beta['a'] = 0.0
        self.beta['o'] = 0.0

        # State bounds
        self.aMax = 11.0
        self.wMax = 3.0

        self.collisionR = collisionR

        # Scale for output normalization 
        # TODO FIX THIS
        self.norm_to = 0.02
        self.mean = 6.0
        self.var = 7.0
        # Set the seed


        self.N_src_samples = num_src_samples
        self.N_boundary_pts = self.N_src_samples//2

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 
        torch.manual_seed(seed)

    def __len__(self):
        return 1
    
    def time_control(self, time, set1, set2):
        time_control = (1 - time)*set1 + (time)*set2
        #time_control = torch.where(time > 1.0, set2, time_control)
        return time_control

    def compute_lx(self, state_coords_unnormalized):
        dist_R1 = torch.norm(state_coords_unnormalized[:, 0:2], dim=1, keepdim=True) - self.collisionR
        return dist_R1
    
    def compute_IC(self, state_coords):
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[:, 0] = state_coords_unnormalized[:, 0] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 1] = state_coords_unnormalized[:, 1] * self.alpha['y'] + self.beta['y']
        
        lx = self.compute_lx(state_coords_unnormalized)
        return lx

    def compute_overall_ham(self, x, dudx, return_components=False):
        alpha = self.alpha
        beta = self.beta

        # Scale the costates appropriately.
        dudx[..., 0] = dudx[..., 0] / alpha['x']
        dudx[..., 1] = dudx[..., 1] / alpha['y']
        dudx[..., 2] = dudx[..., 2] / alpha['th']
        dudx[..., 3] = dudx[..., 3] / alpha['v']
        dudx[..., 4] = dudx[..., 4] / alpha['v']

        
        # Scale the states appropriately.
        x_u = x * 1.0
        x_u[..., 1] = x_u[..., 1] * alpha['x'] + beta['x'] #xrel
        x_u[..., 2] = x_u[..., 2] * alpha['y'] + beta['y'] # yrel
        x_u[..., 3] = x_u[..., 3] * alpha['th'] + beta['th'] # theta rel
        x_u[..., 4] = x_u[..., 4] * alpha['v'] + beta['v'] # v r
        x_u[..., 5] = x_u[..., 5] * alpha['v'] + beta['v'] # v h

        x_u[..., 6] = x_u[..., 6] * alpha['a'] + beta['a']
        x_u[..., 7] = x_u[..., 7] * alpha['a'] + beta['a']
        x_u[..., 8] = x_u[..., 8] * alpha['o'] + beta['o']
        x_u[..., 9] = x_u[..., 9] * alpha['o'] + beta['o']
        x_u[..., 10] = x_u[..., 10] * alpha['a'] + beta['a']
        x_u[..., 11] = x_u[..., 11] * alpha['a'] + beta['a']
        x_u[..., 12] = x_u[..., 12] * alpha['o'] + beta['o']
        x_u[..., 13] = x_u[..., 13] * alpha['o'] + beta['o']
        

        # linear interpolation backwards since BRT
        amin = self.time_control(x_u[...,0], x_u[..., 10], x_u[..., 6])
        amax = self.time_control(x_u[...,0], x_u[..., 11], x_u[..., 7])
        omin = self.time_control(x_u[...,0], x_u[..., 12], x_u[..., 8])
        omax = self.time_control(x_u[...,0], x_u[..., 13], x_u[..., 9])

        # pursuer wants to minimize distance
        o_opt_H = torch.where(dudx[...,2]>0, omin, omax)
        a_opt_H = torch.where(dudx[...,4]>0, amin, amax) 

        ham1 = self.wMax * torch.abs(dudx[..., 0] * x_u[..., 2] - dudx[..., 1] * x_u[..., 1] - dudx[..., 2])  # Control component
        ham2 = self.aMax * torch.abs(dudx[..., 3])
        ham3 = -x_u[..., 4]*dudx[...,0] 

        ham4 = o_opt_H * dudx[..., 2]  # Disturbance component
        ham5 = a_opt_H * dudx[..., 4]
        ham6 = (x_u[...,5] * (torch.cos(x_u[...,3]) * dudx[..., 0])) + (x_u[...,5] * torch.sin(x_u[..., 3]) * dudx[..., 1])
       
        ham = ham1+ham2+ham3+ham4+ham5+ham6
        return ham

    
    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        # Make sure that the oMax and aMax are greater than oMin and aMin respectively. 
        # Shifting the sampling to [aMin, 1] and [oMin, 1] for aMax and oMax respectively
        coords[..., 6] =  (0.5 * (1 - coords[..., 5]) * coords[..., 6]) + (0.5 * (1 + coords[..., 5]))
        coords[..., 10] = (0.5 * (1 - coords[..., 9]) * coords[..., 10]) + (0.5 * (1 + coords[..., 9]))
        coords[..., 8] = (0.5 * (1 - coords[..., 7]) * coords[..., 8]) + (0.5 * (1 + coords[..., 7])) 
        coords[..., 12] = (0.5 * (1 - coords[..., 11]) * coords[..., 12]) + (0.5 * (1 + coords[..., 11]))



        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # Sample some points to impose the boundary coditions
        if self.periodic_boundary:
            # import ipdb; ipdb.set_trace()
            coords_angle = torch.zeros(self.N_boundary_pts, 1).uniform_(math.pi-0.001, self.alpha['th']) # Sample near the right boundary
            coords_angle[0:self.N_boundary_pts//2] = -1.0 * coords_angle[0:self.N_boundary_pts//2] # Assign half of the points to the left boundary
            coords_angle_periodic = angle_normalize(coords_angle)
            coords_angle_concatenated = torch.cat((coords_angle, coords_angle_periodic), dim=0)
            coords_angle_concatenated_normalized = (coords_angle_concatenated)/self.alpha['th']
            coords[:self.N_boundary_pts] = coords[self.N_boundary_pts:2*self.N_boundary_pts]
            coords[:2*self.N_boundary_pts, self.angle_index] = coords_angle_concatenated_normalized[..., 0]  


        # Compute the initial value function
        if self.diffModel:
            coords_var = torch.tensor(coords.clone(), requires_grad=True)
            boundary_values = self.compute_IC(coords_var[..., 1:])
            
            # Normalize the value function
            #print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            #print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))

            # Compute the gradients of the value function
            lx_grads = diff_operators.gradient(boundary_values, coords_var)[..., 1:6]
        else:
            boundary_values = self.compute_IC(coords[..., 1:])

            # Normalize the value function
            # print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            # print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        
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
