import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

#generate random finite 2D lattice with q possible spin types
def generate_random_qpott(height = 8, width = 8, q_val = 3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(0, q_val, size=(height, width)) 

def get_dH(lattice, trial_location_i, trial_location_j, trial_color = None):
    height, width = lattice.shape

    comparison_list = []
    current_color = lattice[trial_location_i, trial_location_j] # returns the number of the color
    comparison_list.append(current_color)

    if trial_color != None:
        comparison_list.append(trial_color)

    energy_list = []

    for number, color in enumerate(comparison_list): #check energy for ORIGINAL Color FIRST, THEN TRIAL Color NEXT
        energy = 0

        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):

            #rmb periodic boundary
            ii = (trial_location_i + di) % height
            jj = (trial_location_j + dj) % width

            neighbour_spin = lattice[ii, jj]

            if color == neighbour_spin:
                indicator = 1
                energy += indicator
            else:
                indicator = 0

        energy_list.append(energy)

    if trial_color == None:
        H = energy_list[0]
        return (H, None, None)
    else:
        H, Hflip = energy_list
        return (-H, -Hflip, (-Hflip)-(-H))


def metropolis_hastings_qpott(q_val, lattice, T, nsteps, snapshot_interval, seed=None):

    #REPRODUCIBILITY SEED
    random.seed(seed)
    np.random.seed(seed)

    #GENERAL PARAMETERS
    height, width = lattice.shape
    if T == 0:
        beta = 0
    else:
        beta = 1.0 / T

    #INITIALIZATION OF RECORDS
    samples = [np.copy(lattice)]
    dh_record = [0] #first dH should be 0 bc it's the start
    energy_record = []
    running_min_list = []

    # Normalize the colormap based on the range of lattice values
    norm = Normalize(vmin=0, vmax=q_val-1)

    #Calculate initial config energy
    initial_energy = 0
    for i in range(height):
        for j in range(width):
            H, _, _ = get_dH(lattice, i, j) 
            initial_energy += H
    energy_record.append(initial_energy)

    running_min = initial_energy
    running_min_list.append(running_min)

    acc_energy_diff = {"-4": 0, "-3": 0, "-2": 0, "-1": 0, "0": 0, "1": 0, "2":0, "3": 0, "4": 0}
    all_energy_diff = {"-4": 0, "-3": 0, "-2": 0, "-1": 0, "0": 0, "1": 0, "2":0, "3": 0, "4": 0}

    #RUN ALGORITHM
    for step in range(nsteps):

        #GENERATE RANDOM NEW CONFIGURATION (Lattice position and color)
        i, j, color = random.randint(0, height - 1), random.randint(0, width - 1), random.randint(0, q_val - 1)

        H, Hflip, dH = get_dH(lattice, i, j, color)

        all_energy_diff[str(dH)] += 1

        #Acceptance criterion : CHECK IF NEW CONFIGURATION IS ACCEPTED
        if dH < 0 or random.uniform(0, 1) < math.exp(-beta * dH):
            lattice[i, j] = color #update this condition (ising model meant change spin but qpotts means change color?)
            initial_energy += dH

            #update running minimum
            if initial_energy < running_min:
                running_min = initial_energy

            #RECORD ENERGY DIFFERENCE (dH) OF ACCEPTED CONFIGURATION
            if acc_energy_diff.get(str(dH)) is not None:
                acc_energy_diff[str(dH)] += 1


        #FOR VISUALIZATION
        if (step+1) % snapshot_interval == 0:
            samples.append(np.copy(lattice))
            dh_record.append(dH)
            energy_record.append(initial_energy) #record energy of the move that was accepted
            running_min_list.append(running_min)

    #update these stats over time too?
    print("accepted config energy diff:",acc_energy_diff)
    print("all proposed energy diff:", all_energy_diff)
    up_moves = ["1", "2", "3", "4"]
    print("up-move acceptance:", sum(acc_energy_diff[key] for key in up_moves)/nsteps)
    print("acceptance probability:", sum(acc_energy_diff.values())/nsteps)


#     VISUALIZATION CODE
    num_samples = len(samples)
    num_cols = 6
    num_rows = (num_samples + num_cols - 1) // num_cols

    fig_width = 15
    fig_height = num_rows * 3

    plt.figure(figsize=(fig_width, fig_height))
    for i, lattice in enumerate(samples):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(lattice, cmap='coolwarm', interpolation='nearest', norm=norm) 
        plt.title(f'Step {i * snapshot_interval}\n dH = {dh_record[i]}, ENERGY = {energy_record[i]}')
        plt.xlabel("running min: " + str(running_min_list[i]))

    plt.tight_layout()
    plt.show()

    return samples, dh_record, energy_record, running_min_list 

## Acceptance Probabilities
def linf_acceptance_probability(c_val, H_x, H_y, T):
    if T == 0:
        beta = 0
    else:
        beta = 1.0 / T
    if H_y <= H_x:
        probability = 1
    elif c_val >= H_y and H_y > H_x:
        probability = math.exp(-beta * (H_y-H_x))
    elif H_y > c_val and c_val >= H_x:
        probability = (math.exp(-beta *(c_val-H_x)))*(T/(H_y - c_val + T))
    elif H_y > H_x and H_x > c_val:
        probability = (H_x - c_val + T)/(H_y - c_val + T)
    return probability

def quadf_acceptance_probability(c_val, H_x, H_y, T):
    if T == 0:
        beta = 0
    else:
        beta = 1.0 / T
    if H_y <= H_x:
        probability = 1
    elif c_val >= H_y > H_x:
        probability = math.exp(-beta * (H_y-H_x))
    elif H_y > c_val >= H_x:
        probability = math.exp(-beta * (c_val-H_x) - (math.sqrt(beta) * math.atan(math.sqrt(beta)*(H_y - c_val))))
    elif H_y > H_x > c_val:
        probability = math.exp(math.sqrt(beta)*(math.atan(math.sqrt(beta)*(H_x-c_val)))) - math.atan(math.sqrt(beta)*(H_y-c_val))
    return probability

def sqrtf_acceptance_probability(c_val, H_x, H_y, T):
    if T == 0:
        beta = 0
    else:
        beta = 1.0 / T
    if H_y <= H_x:
        probability = 1
    elif c_val >= H_y > H_x:
        probability = math.exp(-beta * (H_y-H_x))
    elif H_y > c_val >= H_x:
        probability = math.exp(-beta *(c_val-H_x) - (2*math.sqrt(H_y - c_val))) * (((math.sqrt(H_y-c_val) + T)/T)**(2*T))
    elif H_y > H_x > c_val:
        probability = math.exp(2*math.sqrt(H_x - c_val)-2*math.sqrt(H_y - c_val)) * (((math.sqrt(H_y-c_val) + T)/(math.sqrt(H_x-c_val) + T))**(2*T))

    return probability


## --------- Obtain Acceptance Probability based on energy model ----------
def calc_acceptance_prob(energy_model, c_val, H_x, H_y, dH, T):
    if T == 0:
        beta = 0
    else:
        beta = 1.0 / T

    if energy_model == "original":
        acceptance_probability = np.exp(-beta * dH)
    elif energy_model == "lin":
        acceptance_probability = linf_acceptance_probability(c_val, H_x, H_y, T)
    elif energy_model == "quad":
        acceptance_probability = quadf_acceptance_probability(c_val, H_x, H_y, T)
    elif energy_model == "sqrt":
        acceptance_probability = sqrtf_acceptance_probability(c_val, H_x, H_y, T)

    return acceptance_probability

def modified_MH(q_val, energy_model, lattice, T, nsteps, fixed_val = None, switch_period = None, snapshot_interval = 100, seed=None):

    #REPRODUCIBILITY SEED
    random.seed(seed) # this seed keeps the set of proposed moves the same
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
            H, _, _ = get_dH(lattice, i, j)
            initial_energy += H

    energy_record.append(initial_energy)

    running_min = initial_energy
    running_min_list.append(running_min)

    acc_energy_diff = {"-4": 0, "-3": 0, "-2": 0, "-1": 0, "0": 0, "1": 0, "2":0, "3": 0, "4": 0}
    all_energy_diff = {"-4": 0, "-3": 0, "-2": 0, "-1": 0, "0": 0, "1": 0, "2":0, "3": 0, "4": 0}
    up_moves = ["1", "2", "3", "4"]

    #RUN ALGORITHM
    for step in range(nsteps):
        #GENERATE RANDOM NEW CONFIGURATION
        i, j, color = random.randint(0, height - 1), random.randint(0, width - 1), random.randint(0, q_val - 1)
        H, Hflip, dH = get_dH(lattice, i, j, color)

        all_energy_diff[str(dH)] += 1

        #total energy configurations
        H_x = initial_energy
        H_y = initial_energy + dH


## DEFINING THE C_VAL PARAMETER --------------------------
        if switch_period != None:
            if step <= switch_period: #set number of steps parameter, should be within bounds.
                c_val = running_min
            else:
                c_val = fixed_val
        else:
            c_val = fixed_val

## ADJUST ACCEPTANCE PROBABILITY BASED ON ENERGY MODEL.
        acceptance_probability = calc_acceptance_prob(energy_model, c_val, H_x, H_y, dH, T)

        #Acceptance criterion : CHECK IF NEW CONFIGURATION IS ACCEPTED
        if dH < 0 or random.uniform(0, 1) < acceptance_probability:
            lattice[i, j] = color
            initial_energy = H_y

            #update running minimum
            if H_y < running_min:
                running_min = H_y

            #RECORD ENERGY DIFFERENCE (dH) OF ACCEPTED CONFIGURATION
            if acc_energy_diff.get(str(dH)) is not None:
                acc_energy_diff[str(dH)] += 1

        #FOR VISUALIZATION
        if (step+1) % snapshot_interval == 0:
            samples.append(np.copy(lattice))
            dh_record.append(dH)
            energy_record.append(initial_energy) #record energy of the move that was accepted
            running_min_list.append(running_min)


    print("accepted config energy diff:", acc_energy_diff)
    print("all proposed energy diff:", all_energy_diff)
    print("up-move acceptance:", sum(acc_energy_diff[key] for key in up_moves)/nsteps)
    print("acceptance probability:", sum(acc_energy_diff.values())/nsteps)
    print("algorithm converges: ", converges)
    print("not converged", global_min_col)

    visualization(q_val, samples, dh_record, energy_record, running_min_list, snapshot_interval)

    return samples, dh_record, energy_record, running_min_list
