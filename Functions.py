
import numpy as np
import qutip as qt
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
def time_dependent_hamiltonian(fR, f0, signal_function):
    """

    Arguments:
    fR --> Rabi frequency (Hz)
    f0 --> Larmor frequency (Hz)
    signal_function --> a function representing the external signal

    Returns:
    A time-dependent Hamiltonian compatible with QuTiP's time evolution simulators.
    """

    # Static part of the Hamiltonian based on the Larmor frequency
    H0 =  np.pi* f0 * qt.sigmaz()

    # Time-dependent part of the Hamiltonian based on the Rabi frequency
    H1 = 2*np.pi*fR *qt.sigmax()

    # The total Hamiltonian
    H = [H0, [H1, signal_function]]

    return H
def apply_rotating_frame_on_z(result, rotation_frequency, trange):
    """
    Apply the rotating frame transformation specifically to the z-axis measurement results stored in a QuTiP Result object.

    Arguments:
    result -- QuTiP Result object containing the simulation outcomes
    rotation_frequency -- frequency of the rotating frame
    trange -- array of time points for the simulation

    Returns:
    transformed_results -- list of arrays with results transformed into the rotating frame
    """
    omega = -2 * np.pi * rotation_frequency
    sigma_x_prime = []
    sigma_y_prime = []

    for (pos,t) in enumerate(trange):
        cos_factor = np.cos(omega * t)
        sin_factor = np.sin(omega * t)
        # Assuming result.expect[0] is sigma_x and result.expect[1] is sigma_y
        sigma_x_prime.append(result.expect[0][pos] * cos_factor - result.expect[1][pos] * sin_factor)
        sigma_y_prime.append(result.expect[0][pos] * sin_factor + result.expect[1][pos] * cos_factor)
    sigma_x_prime_array = np.array(sigma_x_prime)
    sigma_y_prime_array = np.array(sigma_y_prime)    
    #print(len(sigma_x_prime_array), len(result.expect[2]), len(sigma_y_prime_array))
    return [sigma_x_prime_array, sigma_y_prime_array, result.expect[2]]

def signal_generator(signal_type="cos", frequency=1e9, amplitude=1, time_array=None):
    "Creates a signal as an array ov values, for a given frequency and signal type"
    if time_array is None:
        raise ValueError("time_array must be provided")
    
    if signal_type == "sin":
        return amplitude * np.sin(2 * np.pi * frequency * time_array)
    elif signal_type == "cos":
        return amplitude * np.cos(2 * np.pi * frequency * time_array)
    elif signal_type == "id":
        return amplitude*time_array
    else:
        raise ValueError("Unsupported signal type. Supported types are 'sin' and 'cos'")
        
def calculate_fidelity(operation, ideal):
    """
    Calculate the process fidelity between the operation and the ideal transformation.
    """
    return np.abs(np.trace((ideal.dag() * operation).full()))**2 / (operation.dim[0][0] * ideal.dim[0][0])
def process_fidelity():
    
    return 
def time_evolution(Hamiltonian, state, time, operators):
    full_evolution= qt.sesolve(Hamiltonian, state, time, operators, progress_bar =None )
    return full_evolution
