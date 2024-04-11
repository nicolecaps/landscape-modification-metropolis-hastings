import random
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def generate_random_initial_lattice(size = 8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return 2 * np.random.randint(0, 2, size=(size, size)) - 1

def get_dH(lattice, trial_location):
    i, j = trial_location #tuple containing the coordinates of the spin considered for a flip.
    height, width = lattice.shape
    H, Hflip = 0, 0
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ii = (i + di) % height
        jj = (j + dj) % width
        H -= lattice[ii, jj] * lattice[i, j]
        Hflip += lattice[ii, jj] * lattice[i, j]
    return H, Hflip, Hflip-H #difference between energy after the flip and energy before the flip.

def metropolis_hastings_ising(lattice, T, nsteps, snapshot_interval, seed=None):
    #REPRODUCIBILITY SEED
    random.seed(seed)
    np.random.seed(seed)

    #GENERAL PARAMETERS
    height, width = lattice.shape
    beta = 1.0 / T

    #INITIALIZATION OF RECORDS
    samples = [np.copy(lattice)]
    dh_record = [0] #first dH should be 0 bc it's the start
    energy_record = []
    running_min_list = []

    #Calculate initial config energy
    initial_energy = 0
    for i in range(height):
        for j in range(width):
            H, _, _ = get_dH(lattice, (i, j))
            initial_energy += H
    energy_record.append(initial_energy)

    running_min = initial_energy
    running_min_list.append(running_min)

    acc_energy_diff = {"-8": 0, "-4": 0, "0": 0, "4": 0, "8": 0}
    all_energy_diff = {"-8": 0, "-4": 0, "0": 0, "4": 0, "8": 0}

    #RUN ALGORITHM
    for step in range(nsteps):
        #GENERATE RANDOM NEW CONFIGURATION
        i, j = random.randint(0, height - 1), random.randint(0, width - 1)

        H, Hflip, dH = get_dH(lattice, (i, j))

        all_energy_diff[str(dH)] += 1

        #Acceptance criterion : CHECK IF NEW CONFIGURATION IS ACCEPTED
        if dH < 0 or random.uniform(0, 1) < np.exp(-beta * dH):
            lattice[i, j] = -lattice[i, j]
            initial_energy += dH

            #update running minimum (for visualization)
            if initial_energy < running_min:
                running_min = initial_energy

            #RECORD ENERGY DIFFERENCE (dH) OF ACCEPTED CONFIGURATION
            if acc_energy_diff.get(str(dH)) is not None:
                acc_energy_diff[str(dH)] += 1

        #FOR VISUALIZATION
        if (step+1) % snapshot_interval == 0:
            samples.append(np.copy(lattice))
            dh_record.append(-dH)
            energy_record.append(initial_energy) #record energy of the move that was accepted
            running_min_list.append(running_min)
          
    print("accepted config energy diff:", acc_energy_diff)
    print("all proposed energy diff:", all_energy_diff)
    up_moves = ["4", "8"]
    print("acceptance probability:", sum(acc_energy_diff[key] for key in up_moves)/nsteps)

#     VISUALIZATION CODE
    num_samples = len(samples)
    num_cols = 6
    num_rows = (num_samples + num_cols - 1) // num_cols

    fig_width = 15
    fig_height = num_rows * 3

    norm = Normalize(vmin=-1, vmax=1)

    plt.figure(figsize=(fig_width, fig_height))
    for i, lattice in enumerate(samples):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(lattice, cmap='gray', interpolation='nearest', norm=norm) #'bwr','gray', 'coolwarm'
        plt.title(f'Step {i * snapshot_interval}\n dH = {dh_record[i]}, ENERGY = {energy_record[i]}')
        plt.xlabel("running min: " + str(running_min_list[i]))

    plt.tight_layout()
    plt.show()

    return samples, dh_record, energy_record, running_min_list #returns the final config
