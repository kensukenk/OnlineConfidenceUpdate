import torch
import torch.nn.functional as F

import diff_operators
import modules, utils

import math
import numpy as np
from scipy.stats import vonmises

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from stats import *


def initialize_hji_MultiVehicleCollisionNE(dataset, minWith):
    # Initialize the loss function for the multi-vehicle collision avoidance problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    numEvaders = dataset.numEvaders
    num_pos_states = dataset.num_pos_states
    alpha_angle = dataset.alpha_angle
    alpha_time = dataset.alpha_time

    def hji_MultiVehicleCollision(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]

            # Scale the costate for theta appropriately to align with the range of [-pi, pi]
            dudx[..., num_pos_states:] = dudx[..., num_pos_states:] / alpha_angle

            # Compute the hamiltonian for the ego vehicle
            ham = velocity*(torch.cos(alpha_angle*x[..., num_pos_states+1]) * dudx[..., 0] + torch.sin(alpha_angle*x[..., num_pos_states+1]) * dudx[..., 1]) - omega_max * torch.abs(dudx[..., num_pos_states])

            # Hamiltonian effect due to other vehicles
            for i in range(numEvaders):
                theta_index = num_pos_states+1+i+1
                xcostate_index = 2*(i+1)
                ycostate_index = 2*(i+1) + 1
                thetacostate_index = num_pos_states+1+i
                ham_local = velocity*(torch.cos(alpha_angle*x[..., theta_index]) * dudx[..., xcostate_index] + torch.sin(alpha_angle*x[..., theta_index]) * dudx[..., ycostate_index]) + omega_max * torch.abs(dudx[..., thetacostate_index])
                ham = ham + ham_local

            # Effect of time factor
            ham = ham * alpha_time

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_MultiVehicleCollision


def initialize_hji_air3D(dataset, minWith):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    alpha_angle = dataset.alpha_angle

    def hji_air3D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]

        x_theta = x[..., 3] * 1.0

        # Scale the costate for theta appropriately to align with the range of [-pi, pi]
        dudx[..., 2] = dudx[..., 2] / alpha_angle
        # Scale the coordinates
        x_theta = alpha_angle * x_theta

        # Air3D dynamics
        # \dot x    = -v_a + v_b \cos \psi + a y
        # \dot y    = v_b \sin \psi - a x
        # \dot \psi = b - a

        # Compute the hamiltonian for the ego vehicle
        ham = omega_max * torch.abs(dudx[..., 0] * x[..., 2] - dudx[..., 1] * x[..., 1] - dudx[..., 2])  # Control component
        ham = ham - omega_max * torch.abs(dudx[..., 2])  # Disturbance component
        ham = ham + (velocity * (torch.cos(x_theta) - 1.0) * dudx[..., 0]) + (velocity * torch.sin(x_theta) * dudx[..., 1])  # Constant component

        # If we are computing BRT then take min with zero
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_air3D

def initialize_hji_human_forward(dataset, minWith):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    velocity = dataset.velocity
    alpha_angle = dataset.alpha_angle
    beta1 = dataset.beta1
    beta2 = dataset.beta2
    goal = dataset.goal

    
    def arcpoint(x,a,b,inner):
        # a and b are the endpoints of the arc,
        # inner denotes whether small arc(true) or big arc (false)
        # returns true if x inside range of [a,b]

        x,a,b = thetaParse(x,a,b)
        
        # a < x < b, b-a is smaller than pi
        one_in = torch.logical_and(torch.logical_and(x < b, (b-a)< np.pi),x-a>0)
        # b-a is larger than pi, x < a or x > b)
        two_in = torch.logical_and((b-a) > np.pi, torch.logical_or(x<a, x>b))
        
        # checking periodicity
        three_in = torch.logical_and(x < a, (x + 2*np.pi) < b)
        four_in = torch.logical_and(x > b, (x - 2*np.pi) > a)  


        # a < x < b, b-a is greater than pi
        one_out = torch.logical_and(torch.logical_and(x < b, (b-a)> np.pi),x-a>0)

        # b-a is larger than pi, x < a or x > b)
        two_out = torch.logical_and((b-a) < np.pi, torch.logical_or(x<a, x>b))
        # checking periodicity
        three_out = torch.logical_and(x < a, (x + 2*np.pi) < (b)) 
        four_out = torch.logical_and(x > b, (x - 2*np.pi) > (a))

        case1 = torch.logical_or(torch.logical_or(torch.logical_or(one_in, two_in), three_in),four_in)
        case2 = torch.logical_or(torch.logical_or(torch.logical_or(one_out, two_out), three_out),four_out)

        return torch.where(inner, case1, case2)
       
    def pointDist(x,a,b):
        # x is point out of range
        # a is lower bound and b is upper bound
        # return true if closer to a, false otherwise
        x, a, b = thetaParse(x,a,b)
        x = x% (2*np.pi)
        a = a% (2*np.pi)
        b = b% (2*np.pi)

        x_a1 = torch.abs(x - a)
        x_b1 = torch.abs(x - b)
        x_a2 = 2*np.pi - x_a1
        x_b2 = 2*np.pi - x_b1

        x_a = torch.where(x_a1<x_a2, x_a1, x_a2)
        x_b = torch.where(x_b1<x_b2, x_b1, x_b2)

        
        return torch.where(x_a < x_b, True, False)
  

    def thetaParse(x,a,b):
        #a = torch.where(a< -np.pi, a + 2*np.pi, a)
        #a = torch.where(a> np.pi, a - 2*np.pi, a)
        #b = torch.where(b< -np.pi, b + 2*np.pi, b)
        #b = torch.where(b> np.pi, b - 2*np.pi, b)
        x = x + 2*np.pi
        a = a + 2*np.pi
        b = b + 2*np.pi
        b = torch.where(b<a, b+2*np.pi, b)
        #temp = torch.clone(a)
        #a = torch.where(b<a, b,a)
        #b = torch.where(b==a, temp, b)
        
        #b = b - a
        #x = x - a
        return x, a, b
 
    def hji_human_forward(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 5)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]

        # human dynamics
        # \dot x    = v \cos u
        # \dot y    = v \sin u
        # \dot x0   = 0
        # \dot y0   = 0
        # \dot p    = 0
        
        u_star = torch.atan2(-goal[1]+x[...,2], -goal[0]+x[...,1]) # angle directly to goal

        # control that maximizes the hamiltonians
        control1 = torch.atan2(dudx[...,1], dudx[...,0])
        control2 = torch.where(control1 > 0, control1 - np.pi, control1 + np.pi)
        
        #beta =  torch.pow(x[...,5],2)*beta1 + torch.pow((1-x[...,5]), 2)*beta2
        beta = x[...,5]*beta1 + (1-x[...,5])*beta2
        # EXPERIMENT spread = np.pi - truncnorm_ppf(0.95,torch.tensor(-np.pi),torch.tensor(np.pi), loc = 0, scale = beta)
        spread = truncnorm_ppf(0.95,torch.tensor(-np.pi),torch.tensor(np.pi), loc = 0, scale = beta)

        umin = u_star - spread
        umax = u_star + spread
        
        offset1 = torch.where(pointDist(control1,umin,umax), umin, umax)
        control1 = torch.where(arcpoint(control1,umin,umax,inner = spread<np.pi/2), control1, offset1)

  
        offset2 = torch.where(pointDist(control2,umin,umax), umin,umax)
        control2 = torch.where(arcpoint(control2,umin,umax,inner = spread<np.pi/2), control2, offset2)

        # human dynamics
        # \dot x    = v \cos u
        # \dot y    = v \sin u
        # \dot x0   = 0
        # \dot y0   = 0
        # \dot p    = 0
        
        ham = dudx[...,0]*velocity*(torch.cos(control1)) + dudx[...,1]*velocity*(torch.sin(control1))
        ham2 = dudx[...,0]*velocity*(torch.cos(control2)) + dudx[...,1]*velocity*(torch.sin(control2))
        ham = torch.where(ham2<ham, ham2, ham)
        
       
        # If we are computing BRT then take min with zero
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_human_forward