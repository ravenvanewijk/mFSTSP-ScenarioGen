import numpy as np
import matplotlib.pyplot as plt

def generate_delivery_times(n, desired_mu_delay=10, min_delay=-10, 
                            seed=11, shape_parameter=3):
    """
    Generates a sample of delivery times following a Gamma distribution, with a fixed seed for reproducibility.

    Parameters:
    - n (int): The number of delivery times to generate.
    - mean_delivery_time (float): The average (mean) delivery time.
    - seed (int): The seed for the random number generator.
    - shape_parameter (float): The shape parameter (k) of the Gamma distribution.
    
    Returns:
    - np.ndarray: Array of generated delivery times.
    """
    mu_dist = desired_mu_delay - min_delay
    np.random.seed(seed)  # Set the seed for reproducibility
    scale_parameter = mu_dist / shape_parameter  # θ = μ / k
    delay_times = np.random.gamma(shape=shape_parameter, scale=scale_parameter, size=n)
    return delay_times + min_delay

def generate_drone_speed(prob, mag, length=1):
    """This function randomly determines whether a drone travels at its usual 
    speed, slower or faster. 
    
    Arguments:
        - prob: float, probability of mutation (between 0 and 1)
        - mag: float, percentage change of the speed of the drone if 
          the speed is changed
        - length: int, number of speeds to randomize
    """
    factor_lst = []
    
    for _ in range(length):
        if np.random.rand() < prob:
            factor = np.random.choice([1 - mag, 1 + mag])
        else:
            factor = 1
        factor_lst.append(float(factor))
    
    return factor_lst

def plot_delivery_times(delivery_times, mean_delivery_time):
    """
    Plots the histogram of the delivery times.

    Parameters:
    - delivery_times (np.ndarray): Array of generated delivery times.
    - mean_delivery_time (float): The average (mean) delivery time.
    """
    plt.hist(delivery_times, bins=30, density=True, alpha=0.7, color='green')
    plt.xlabel('Total Delivery Time (seconds)')
    plt.ylabel('Probability Density')
    plt.title(f'Gamma Distribution of Delivery Times\nMean = {mean_delivery_time} seconds')
    plt.grid(True)
    plt.show()

# de = generate_delivery_times(2500)
# plot_delivery_times(de, 10)

uncertainty_settings = {
            'light': {'mu_del': 10, 'min_delay': -10,
                        'spd_change_prob': 0.2, 'spd_change_mag': 0.2,
                        'stop_length': 30, 'stop_interval': 360},
            'medium': {'mu_del': 30, 'min_delay': -15,
                        'spd_change_prob': 0.5, 'spd_change_mag': 0.3,
                        'stop_length': 30, 'stop_interval': 240},
            'heavy': {'mu_del': 60, 'min_delay': -20,
                        'spd_change_prob': 0.8, 'spd_change_mag': 0.4,
                        'stop_length': 30, 'stop_interval': 120}
                        }
