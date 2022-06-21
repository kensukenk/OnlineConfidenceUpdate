import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import diff_operators
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import modules, dataio
import torch

model = modules.SingleBVPNet(in_features=6, out_features=1, final_layer_factor=1.,hidden_features=512, num_hidden_layers=3)
model.cuda()
model.load_state_dict(torch.load('logs/betaVary_pre50_src15_epo200_rad02_betas01_2/checkpoints/model_final.pth'))

# Time values at which the function needs to be plotted
times = [0., 0.1*(1 - 0.1), 0.25*(1- 0.1), 0.75*(1 - 0.1), 1 - 0.1]
num_times = len(times)
# xy slices to be plotted
probs = [0.55, 0.65, 0.75, 0.85, 0.95]
num_probs = len(probs)
# Create a figure
fig = plt.figure(figsize=(5*num_times, 5*num_probs))
# Get the meshgrid in the (x, y) coordinate
sidelen = 200
mgrid_coords = dataio.get_mgrid(sidelen)

# Start plotting the results
for i in range(num_times):
  time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]
  x_coords = torch.ones(mgrid_coords.shape[0], 1) * -0.5
  y_coords = torch.ones(mgrid_coords.shape[0], 1) * 0.5
  for j in range(num_probs):
    p_coords = torch.ones(mgrid_coords.shape[0], 1) * probs[j]
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
    ax = fig.add_subplot(num_times, num_probs, (j+1) + i*num_probs)
    ax.set_title('t = %0.2f, p = %0.2f' % (times[i], probs[j]))
    s = ax.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
    print(model_out.T)
    plt.scatter(0, 0)
    fig.colorbar(s) 

fig.savefig( 'finalBetaVary.png')



