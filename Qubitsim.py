
import numpy as np
import qutip as qt
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def single_spin_hamiltonian(fR, f0, mw_signal, rotating_frame=None ):
    """
    Arguments:
    fR --> Rabi frequency (Hz)
    f0 --> Larmor frequency (Hz)
    mw_signal --> a function representing the external signal

    Returns:
    A time-dependent Hamiltonian compatible with QuTiP's time evolution simulators.
    """

    # Static part of the Hamiltonian based on the Larmor frequency
    H0 =  np.pi* f0 * qt.sigmaz()

    # Time-dependent part of the Hamiltonian based on the Rabi frequency
    H1 = 2*np.pi*fR *qt.sigmax()

    # The total Hamiltonian
    H = [H0, [H1, mw_signal]]

    return H


def rwa_hamiltonian( f0 ):
  """
  Arguments:
  fR --> Rabi frequency (Hz)
  f0 --> Larmor frequency (Hz)
  mw_signal --> a function representing the external signal

  Returns:
  A time-dependent Hamiltonian compatible with QuTiP's time evolution simulators.
  """

  # Static part of the Hamiltonian based on the Larmor frequency
  H0 =  -np.pi* f0 * qt.sigmaz()


  # The total Hamiltonian
  H = H0

  return H

def double_spin_hamiltonian( f0, J_signal, rotating_frame=None ):
    """
    Arguments:
    f0 --> Larmor frequency list (Hz)
    J_signal --> a function representing the external signal
    Returns:
    A time-dependent Hamiltonian compatible with QuTiP's time evolution simulators.
    """

    # Static part of the Hamiltonian based on the Larmor frequency
    Ez = 2*np.pi*(f0[0]+f0[1])/2;
    dEz = 2*np.pi*(f0[0]-f0[1]);

    H0 = qt.Qobj([[Ez,0,0,0],[0,dEz/2,0,0],[0,0,-dEz/2,0],[0,0,0,-Ez]]);

    # Time-dependent part of the Hamiltonian based on the Rabi frequency
    H1 = 2*np.pi*qt.Qobj([[0,0,0,0],[0,-1,1,0],[0,1,-1,0],[0,0,0,0]])/2;

    # The total Hamiltonian
    H = [H0, [H1, J_signal]]

    return H


def rwa_hamiltonian_2qubit( f0 ):
    """
    Arguments:
    fR --> Rabi frequency (Hz)
    f0 --> Larmor frequency (Hz)
    mw_signal --> a function representing the external signal

    Returns:
    A time-dependent Hamiltonian compatible with QuTiP's time evolution simulators.
    """

    # Static part of the Hamiltonian based on the Larmor frequency
    H0 =  -np.pi* f0[0] * qt.sigmaz()
    H1 =  -np.pi* f0[1] * qt.sigmaz()

    #H = qt.tensor(H0, qt.identity(2)) + qt.tensor(qt.identity(2), H1)
    H = qt.Qobj( np.kron(H0, qt.identity(2)) + np.kron(qt.identity(2), H1) )

    return H


def plot_signal(signal, sampling_rate=None, xlim = None):

    trange = np.arange(0, signal.size , 1)*sampling_rate

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(trange/1e-9, signal)
    ax.legend()
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude')
    if xlim is not None:
      plt.xlim(xlim)
    fig.show()

def plot_fft(signal, sampling_rate=None, xlim = None):
  """
  Calculates and plots the FFT of a signal.

  Args:
    signal: A list or NumPy array containing the signal data.
    sampling_rate: The sampling rate of the signal (optional).
    title: The title for the plot (optional).
  """

  # Zero padding
  padded_signal = np.pad(signal, (0, 10*len(signal)), mode='constant')

  # Compute FFT
  fft = np.fft.fft(signal)

  # Absolute values for magnitude
  mag = 20*np.log10(np.abs(fft))

  # Frequencies
  if sampling_rate is not None:
    freqs = np.fft.fftfreq(len(signal), sampling_rate)
  else:
    freqs = np.fft.fftfreq(len(signal))

  # Half the spectrum for real signals
  if np.isrealobj(signal):
    mag = mag[:len(mag)//2]
    freqs = freqs[:len(freqs)//2]

  # Plot
  plt.figure()
  plt.plot(freqs, mag)
  if xlim is not None:
    plt.xlim(xlim)
  plt.xlabel("Frequency" if sampling_rate is not None else "Normalized Frequency")
  plt.ylabel("Signal Power (dB)")
  plt.title("Signal FFT")
  plt.show()

def calculate_fidelity(U, U_ideal):
    """
    Calculate the process fidelity between the operation and the ideal transformation.
    """
    dim = U.shape[0]*U.shape[1]
    return np.abs(np.trace((U.dag() * U_ideal).full()))**2/(dim)

def single_qubit_evolution(larmor_frequency, rabi_frequency, signal_array, trange, initial_state, target_operator, plot2D=False, plot3D=False, index=1, RWA = False):

  # Create the time-dependent Hamiltonian
  Hamiltonian = single_spin_hamiltonian(fR=rabi_frequency, f0=larmor_frequency, mw_signal=signal_array)
  Hamoltonian_rwa = rwa_hamiltonian(larmor_frequency);

  # Calculate the operator with and w/o rotating frame approx.
  # qt.propagator returns list of U for each time step
  U = qt.propagator(Hamiltonian,trange);
  U_w_rwa = qt.propagator(Hamoltonian_rwa,trange)*U;

  # Fidelity can be calculated with
  fidelity = calculate_fidelity( U_w_rwa[-1], target_operator );

  if ( plot2D | plot3D == True ):

    if (RWA):
      states = U_w_rwa*initial_state;
    else:
      states = U*initial_state;

    meas_basis = [qt.sigmax(),qt.sigmay(),qt.sigmaz()]

    if( plot2D == True):
      fig, ax = plt.subplots(figsize=(8,3))
      ax.plot(trange/1e-9, qt.expect(meas_basis[2],states), label="Sigma z measurement")
      ax.plot(trange/1e-9, qt.expect(meas_basis[0],states), label="Sigma x measurement")
      #ax.plot(trange/1e-9, qt.expect(meas_basis[1],states), label="Sigma y measurement")
      ax.legend()
      ax.set_xlabel('Time')
      ax.set_ylabel("Qubit " +str(index+1))
    if( plot3D == True):
      b = qt.Bloch()
      b.add_points([qt.expect(meas_basis[0],states), qt.expect(meas_basis[1],states), qt.expect(meas_basis[2],states)])
      b.size=[2,2]
      b.show()

  return (fidelity)

def double_qubit_evolution(larmor_frequencies, signal_array, trange, initial_state, target_operator, plot2D=False, plot3D=False, RWA=True):

  # Create the time-dependent Hamiltonian
  Hamiltonian = double_spin_hamiltonian( f0=larmor_frequencies, J_signal=signal_array)
  H_rwa = rwa_hamiltonian_2qubit(larmor_frequencies);

  # Calculate the operator with and w/o rotating frame approx.
  # qt.propagator returns list of U for each time step
  U = qt.propagator(Hamiltonian,trange);
  U_rwa = qt.propagator(H_rwa,trange);

  # For CPHASE gate some calibration is needed
  # This can be done by applying two Z gates on the qubits
  # Z gate can be achieved by changing the rotating frame reference therefore no signal needs to be sent
  dEz = np.abs(larmor_frequencies[1] - larmor_frequencies[0])
  dt = trange[1]-trange[0]
  theha_cali = np.sum( dEz - np.sqrt(signal_array**2 + dEz**2)*dt/0.5/2 )
  U_calibration = qt.Qobj(np.array([[1, 0, 0, 0],[0, np.exp(-1j*np.pi*theha_cali), 0, 0],[0, 0, np.exp(1j*np.pi*theha_cali), 0],[0, 0, 0, 1]]))

  # Fidelity calculation
  fidelity = calculate_fidelity( U_calibration*U_rwa[-1]*U[-1], target_operator );

  if ( plot2D | plot3D == True ):

    if (RWA):
      states = U_rwa*U*initial_state;
    else:
      states = U*initial_state;

    if( plot2D == True):
      # Create Measurement basis
      # |1+>
      meas_basis1 = qt.Qobj(np.array([0,0,1,1]/np.sqrt(2)))*qt.Qobj(np.array([0,0,1,1]/np.sqrt(2))).dag()
      # |1->
      meas_basis2 = qt.Qobj(np.array([0,0,1,-1]/np.sqrt(2)))*qt.Qobj(np.array([0,0,1,-1]/np.sqrt(2))).dag()
      # 1-|1->-|1+>
      meas_basis3 = qt.Qobj(np.array([0,1,0,0]))*qt.Qobj(np.array([0,1,0,0])).dag()

      
      # Plot the measurement basis results
      fig, ax = plt.subplots(figsize=(8,3))
      ax.plot(trange/1e-9, qt.expect(meas_basis1,states))
      ax.set_xlabel('Time (ns)')
      ax.set_ylabel(r'$P _{|1+\rangle}$')

      fig, ax = plt.subplots(figsize=(8,3))
      ax.plot(trange/1e-9, qt.expect(meas_basis2,states))
      ax.set_xlabel('Time (ns)')
      ax.set_ylabel(r'$P _{|1-\rangle}$')

      fig, ax = plt.subplots(figsize=(8,3))
      ax.plot(trange/1e-9, qt.expect(meas_basis3,states))
      ax.set_xlabel('Time (ns)')
      ax.set_ylabel(r'$1-(P _{|1+\rangle}+P _{|1-\rangle})$')

    if( plot3D == True):

      # Create Measurement basis
      meas_basis1 = qt.Qobj(np.kron(qt.sigmax(),qt.identity(2)))
      meas_basis2 = qt.Qobj(np.kron(qt.sigmay(),qt.identity(2)))
      meas_basis3 = qt.Qobj(np.kron(qt.sigmaz(),qt.identity(2)))

      nrm = mpl.colors.Normalize(min(trange), max(trange))
      colors = cm.cool(nrm(trange))
      
      # Plot the measurement basis results
      b = qt.Bloch()
      b.add_points([qt.expect(meas_basis1,states), qt.expect(meas_basis2,states), qt.expect(meas_basis3,states)], "m")
      b.point_color = list(colors)
      b.size=[3,3]
      b.show()

      # Create Measurement basis
      meas_basis1 = qt.Qobj(np.kron(qt.identity(2),qt.sigmax()))
      meas_basis2 = qt.Qobj(np.kron(qt.identity(2),qt.sigmay()))
      meas_basis3 = qt.Qobj(np.kron(qt.identity(2),qt.sigmaz()))

      # Plot the measurement basis results
      b = qt.Bloch()
      b.add_points([qt.expect(meas_basis1,states), qt.expect(meas_basis2,states), qt.expect(meas_basis3,states)], "m")
      b.point_color = list(colors)
      b.size=[3,3]
      b.show()

  return (fidelity)
