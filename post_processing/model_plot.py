import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) )
import math
import modules, dataio
import torch

model = modules.SingleBVPNet(in_features=6, out_features=1, final_layer_factor=1.,hidden_features=512, num_hidden_layers=3)
model.cuda()
model.load_state_dict(torch.load('../logs/HR_PE_pre40_src15_epo150_rad02_betas01_2/checkpoints/model_final.pth'))
#model.load_state_dict(torch.load('logs/betaVary_pre50_src15_epo200_rad02_betas01_2/checkpoints/model_final.pth'))

# Time values at which the function needs to be plotted
times = [0., 0.1*(1 - 0.1), 0.25*(1- 0.1), 0.75*(1 - 0.1), 1 - 0.1]
num_times = len(times)
# xy slices to be plotted

thetas = [-math.pi, -0.5*math.pi, 0., 0.5*math.pi, math.pi]
posx = [0.0, 0.5,-0.5, 0.0]
posy = [0.0, 0.5, 0.0, 0.5]
num_pos = len(posx)
num_thetas = len(thetas)
# Create a figure
fig = plt.figure(figsize=(5*num_times, 5*num_pos))
# Get the meshgrid in the (x, y) coordinate
sidelen = 200
mgrid_coords = dataio.get_mgrid(sidelen)

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

fig.savefig( 'finalBetaVary.png')



