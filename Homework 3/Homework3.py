import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    num_samples = 1000000
    d1_samples = []
    d2_samples = []

    # Get samples from d1
    for i in range(num_samples):
        rand = np.random.uniform()
        if rand < 1/3:
            sample = np.random.normal(0,1,1).item()
            d1_samples.append(sample)
        else: 
            sample = np.random.normal(1,1,1).item()
            d1_samples.append(sample)

    # Get samples from d2
    for i in range(num_samples):
        rand = np.random.uniform()
        if rand < 0.5:
            sample = np.random.normal(1/3,1,1).item()
            d2_samples.append(sample)
        else: 
            sample = np.random.normal(1,1,1).item()
            d2_samples.append(sample)
    
    d1_samples = np.array(d1_samples)
    d2_samples = np.array(d2_samples)

    d1_squared = d1_samples**2
    d1_cubed = d1_samples**3
    d1_quad = d1_samples**4

    d2_squared = d2_samples**2
    d2_cubed = d2_samples**3
    d2_quad = d2_samples**4

    # Compute the empirical averages of both d1 and d2
    d1_empiracal_average = np.mean(d1_samples)
    d2_empiracal_average = np.mean(d2_samples)

    # Compute the empircal averages of d1 and d2 ^2, ^3, ^4
    d1_squared_empiracle_average = np.mean(d1_squared)
    d1_cubed_empiracle_average = np.mean(d1_cubed)
    d1_quad_empiracle_average = np.mean(d1_quad)

    d2_squared_empiracle_average = np.mean(d2_squared)
    d2_cubed_empiracle_average = np.mean(d2_cubed)
    d2_quad_empiracle_average = np.mean(d2_quad)
    
    print("The empirical average of d1 is: ")
    print(d1_empiracal_average)
    print("The empirical average of d2 is: ")
    print(d2_empiracal_average)

    print("The empirical average of d1_squared is: ")
    print(d1_squared_empiracle_average)
    print("The empirical average of d2_squared is: ")
    print(d2_squared_empiracle_average)

    print("The empirical average of d1_cubed is: ")
    print(d1_cubed_empiracle_average)
    print("The empirical average of d2_cubed is: ")
    print(d2_cubed_empiracle_average)

    print("The empirical average of d1_quad is: ")
    print(d1_quad_empiracle_average)
    print("The empirical average of d2_quad is: ")
    print(d2_quad_empiracle_average)

    # Plot 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].hist(d1_samples, bins=50, density=True, alpha=0.6, edgecolor='black')
    axes[0].set_title('Mixture D1 = 1/3 N(0,1) + 2/3 N(1,1)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')

    axes[1].hist(d2_samples, bins=50, density=True, alpha=0.6, edgecolor='black')
    axes[1].set_title('Mixture D2 = 1/2 N(1/3,1) + 1/2 N(1,1)')
    axes[1].set_xlabel('x')

    plt.tight_layout()
    plt.show()

