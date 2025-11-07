# Data setup for regression on the Runge function with noise

import autograd.numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def runge_function(x):
    return 1.0 / (1.0 + 25 * x**2)

def make_data(seed_fixed=42, N=200, noise_std=0.1, test_N=2000):
    """
    Generates training, validation, and fixed test datasets based on the Runge function with added Gaussian noise.
    Scales the features using StandardScaler. 2000 test points are generated in the range [-1, 1], 
    for stable evaluation (no fluctuating test set performance).
    """
    # Fixed test set (for final evaluation)
    rng_fixed = np.random.default_rng(seed_fixed)
    X_fixed_test = rng_fixed.uniform(-1, 1, size=(test_N, 1))
    y_fixed_test = runge_function(X_fixed_test) + rng_fixed.normal(0, noise_std, size=(test_N, 1))

    # Training/validation data
    rng = np.random.default_rng(seed_fixed)
    x = rng.uniform(-1, 1, size=(N, 1))
    noise = rng.normal(loc=0.0, scale=noise_std, size=(N, 1))
    y = runge_function(x) + noise

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=42)

    # Scale (fit on train; transform val and fixed test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_fixed_test_scaled = scaler.transform(X_fixed_test)

    
    return (
        seed_fixed,
        rng_fixed,
        X_fixed_test,
        y_fixed_test,
        N,
        rng,
        x,
        noise,
        y,
        X_train,
        X_val,
        y_train,
        y_val,
        scaler,
        X_train_scaled,
        X_val_scaled,
        X_fixed_test_scaled,
    )


