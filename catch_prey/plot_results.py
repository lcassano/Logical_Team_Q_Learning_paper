## Simple script that plots the paper's results ##

import numpy as np
import matplotlib.pyplot as plt

percentage_games_won = {'LTQL': [], 'HystereticQ': [], 'IQL': [], 'Qmix': [], 'Qtran': []}
returns = {'LTQL': [], 'HystereticQ': [], 'IQL': [], 'Qmix': [], 'Qtran': []}
colors = {'LTQL': 'b', 'HystereticQ': 'g', 'IQL': 'c', 'Qmix': 'r', 'Qtran': 'm'}

epochs = 100e3
nmbr_seeds = 2

for agent in percentage_games_won:
  for n in range(nmbr_seeds):
    fname_test = agent + '/results/'+str(n)+'/percentage_games_won_seed_%d.csv' % n
    fname_return = agent + '/results/' + str(n) + '/returns_seed_%d.csv' % n
    win_curve = np.expand_dims(np.loadtxt(fname=fname_test, delimiter=','), axis=0)
    return_curve = np.expand_dims(np.loadtxt(fname=fname_return, delimiter=','), axis=0)
    if n == 0:
      percentage_games_won[agent] = win_curve
      returns[agent] = return_curve
    else:
      percentage_games_won[agent] = np.stack([percentage_games_won[agent], win_curve], axis=0)
      returns[agent] = np.stack([returns[agent], return_curve], axis=0)


fig1, ax1 = plt.subplots()
plt.xlabel('Epochs (thousands)', fontsize=25)
plt.ylabel('Test win %', fontsize=25)
for agent in percentage_games_won:
  curve = 100 * np.squeeze(percentage_games_won[agent])
  test_step = epochs / (percentage_games_won[agent].shape[-1] - 1)
  x = np.arange(start=0, stop=epochs + test_step, step=test_step)/1e3
  ax1.plot(x, np.mean(curve, axis=0), color=colors[agent])
  y_min = np.min(curve, axis=0)
  y_max = np.max(curve, axis=0)
  ax1.fill_between(x, y_min, y_max, alpha=0.3, color=colors[agent])

plt.grid()
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.legend(['LTQL', 'HystQ', 'IQL', 'Qmix','Qtran'], fontsize=15)
fig1.savefig(fname='games_won', bbox_inches='tight')

fig2, ax2 = plt.subplots()
plt.xlabel('Epochs (thousands)', fontsize=25)
plt.ylabel('Average test return', fontsize=25)
for agent in returns:
  curve = np.squeeze(returns[agent])
  test_step = epochs / (returns[agent].shape[-1] - 1)
  x = np.arange(start=0, stop=epochs + test_step, step=test_step)/1e3
  ax2.plot(x, np.mean(curve, axis=0), color=colors[agent])
  y_min = np.min(curve, axis=0)
  y_max = np.max(curve, axis=0)
  ax2.fill_between(x, y_min, y_max, alpha=0.3, color=colors[agent])

plt.grid()
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.legend(['LTQL', 'HystQ', 'IQL', 'Qmix','Qtran'], fontsize=15)
fig2.savefig(fname='returns', bbox_inches='tight')

plt.close('all')