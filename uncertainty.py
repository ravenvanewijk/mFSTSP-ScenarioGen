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
    delivery_times = np.random.gamma(shape=shape_parameter, scale=scale_parameter, size=n)
    return delivery_times + min_delay

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
            'light': {'mu_del': 10, 'min_delay': -10},
            'medium': {'mu_del': 15, 'min_delay': -15},
            'heavy': {'mu_del': 20, 'min_delay': -20},
                        }