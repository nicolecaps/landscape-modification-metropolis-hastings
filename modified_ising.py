## Acceptance Probabilities According to Landscape Modification
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


def modified_MH(energy_model, lattice, T, nsteps, fixed_val = None, switch_period = None, snapshot_interval = 100, seed=None):

    #REPRODUCIBILITY SEED
    random.seed(seed)
    np.random.seed(seed)

    #GENERAL PARAMETERS
    height, width = lattice.shape

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
    up_moves = ["4", "8"]

    #RUN ALGORITHM
    for step in range(nsteps):
        #GENERATE RANDOM NEW CONFIGURATION
        i, j = random.randint(0, height - 1), random.randint(0, width - 1)
        H, Hflip, dH = get_dH(lattice, (i, j))

        all_energy_diff[str(dH)] += 1
        #-----total energy configurations------
        H_x = initial_energy
        H_y = initial_energy + dH

        if switch_period != None:
            if step <= switch_period: #set number of steps parameter, should be within bounds.
                c_val = running_min
            else:
                c_val = fixed_val
        else:
            c_val = fixed_val

        #Acceptance criterion : CHECK IF NEW CONFIGURATION IS ACCEPTED
        acceptance_probability = calc_acceptance_prob(energy_model, c_val, H_x, H_y, dH, T)

        if dH < 0 or random.uniform(0, 1) < acceptance_probability:
            lattice[i, j] = -lattice[i, j]
            initial_energy = H_y

            #update running minimum
            if H_y < running_min:
                running_min = H_y

            # #RECORD ENERGY DIFFERENCE (dH) OF ACCEPTED CONFIGURATION
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
    print("up-move acceptance:", sum(acc_energy_diff[key] for key in up_moves)/nsteps)
    print("acceptance probability:", sum(acc_energy_diff.values())/nsteps)

    #VISUALIZATION CODE
    visualization(samples, dh_record, energy_record, running_min_list, snapshot_interval)

    return samples, dh_record, energy_record, running_min_list #returns the final config
