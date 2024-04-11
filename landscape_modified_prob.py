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
