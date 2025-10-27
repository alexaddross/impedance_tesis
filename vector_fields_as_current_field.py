# Core RLC impedance exploration: calculations + visualizations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi


# -------- Parameters (edit these) --------
R = 50.0          # Ohms
L = 200e-6        # Henry
C = 1e-6          # Farad
V_amp = 1.0       # Source amplitude (phasor magnitude), Volts
    
# Frequency sweep
f_min, f_max, n_points = 10, 1_000_000, 3000  # Hz
f = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
w = 2 * np.pi * f
j = 1j

# -------- Impedance (series RLC) --------
Z = R + j * (w * L - 1 / (w * C))
Z_mag = np.abs(Z)
Z_phase = np.angle(Z, deg=True)

# Current phasor magnitude for given source amplitude
I_amp = V_amp / Z_mag  # |I| = |V| / |Z|

# Power metrics (steady-state sinusoidal; use phasors)
# Apparent power S = V_rms * I_rms; with V_amp as peak magnitude, V_rms = V_amp / sqrt(2)
V_rms = V_amp / np.sqrt(2)
I_rms = I_amp / np.sqrt(2)
S = V_rms * I_rms  # volt-amps
pf = np.cos(np.deg2rad(Z_phase))  # power factor
P = S * pf          # active power (W)
Q = S * np.sin(np.deg2rad(Z_phase))  # reactive power (var)

# Resonant frequency (series RLC), if defined
f0 = 1.0 / (2 * pi * np.sqrt(L * C))
w0 = 2 * pi * f0

# Assemble table
df = pd.DataFrame({
    "f_Hz": f,
    "omega_rad_s": w,
    "ReZ_ohm": np.real(Z),
    "ImZ_ohm": np.imag(Z),
    "absZ_ohm": Z_mag,
    "phase_deg": Z_phase,
    "I_amp_A": I_amp,
    "V_rms_V": np.full_like(f, V_rms),
    "I_rms_A": I_rms,
    "S_VA": S,
    "P_W": P,
    "Q_var": Q,
    "pf": pf
})

# Save for download
csv_path = "rlc_impedance_sweep.csv"
df.to_csv(csv_path, index=False)

# -------- Visualizations --------
# 1) |Z| vs frequency (log-log)
plt.figure()
plt.loglog(f, Z_mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("|Z| (Ohm)")
plt.title("Series RLC: Impedance Magnitude vs Frequency")
plt.grid(True, which="both")
plt.axvline(f0, linestyle="--")
plt.show()

# 2) Phase(Z) vs frequency (semilog-x)
plt.figure()
plt.semilogx(f, Z_phase)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase(Z) (deg)")
plt.title("Series RLC: Impedance Phase vs Frequency")
plt.grid(True, which="both")
plt.axvline(f0, linestyle="--")
plt.show()

# 3) Current magnitude vs frequency
plt.figure()
plt.loglog(f, I_amp)
plt.xlabel("Frequency (Hz)")
plt.ylabel("|I| (A) for |V|=1V")
plt.title("Series RLC: Current Magnitude vs Frequency")
plt.grid(True, which="both")
plt.axvline(f0, linestyle="--")
plt.show()

# 4) Time-domain steady-state at a chosen frequency
f_demo = f0  # choose resonance for illustration
w_demo = 2 * np.pi * f_demo
T_demo = 1 / f_demo
t = np.linspace(0, 3*T_demo, 2000)
# Source u(t) = V_amp * sqrt(2) * cos(wt) if we want peak = V_amp; we already defined phasor as peak magnitude
u_t = V_amp * np.cos(w_demo * t)
# i(t) lags/leads by phase: i(t) = |I| * cos(wt - phase)
# Find nearest frequency index for |I| and phase
idx = (np.abs(f - f_demo)).argmin()
i_t = I_amp[idx] * np.cos(w_demo * t - np.deg2rad(Z_phase[idx]))

plt.figure()
plt.plot(t, u_t, label="u(t) (V)")
plt.plot(t, i_t, label="i(t) (A)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Time-Domain Steady State at f = {f_demo:.1f} Hz")
plt.grid(True)
plt.legend()
plt.show()

summary = {
    "R_ohm": R,
    "L_H": L,
    "C_F": C,
    "V_amp_V": V_amp,
    "f0_Hz (series resonance)": f0,
    "omega0_rad_s": w0,
    "Notes": "Edit R, L, C, V_amp at the top and re-run to explore other cases."
}
