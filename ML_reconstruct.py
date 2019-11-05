import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import pywigxjpf as pywig
from joblib import Parallel, delayed
import sys, argparse, multiprocessing
from common import *

print("  Creating pywig tables...")
NSIDE_WORKING = NSIDE
pywig.wig_table_init(3*NSIDE_WORKING, 3)
pywig.wig_temp_init(3*NSIDE_WORKING)

OUTPUT_DIR = MAPS_OUTPUT_DIR+"ML_reconstruction/"

ML_map = getMap("../ML", NSIDE=NSIDE_WORKING)
ksz_map = getMap("../ksz", NSIDE=NSIDE_WORKING) # websky kSZ
CMB_alms = hp.fitsfunc.read_alm('lensed_alm.fits').astype(np.complex)
CMB_map = hp.alm2map(CMB_alms, NSIDE_WORKING)
Obs_T_map = ML_map + CMB_map


# Temperature alm's, Cl's
almT = hp.map2alm(Obs_T_map)
ClTT = hp.anafast(Obs_T_map)
ls = np.arange(ClTT.size)
norm = ls*(ls+1)

# Read in projected density map, integrate over small radial bins
delta = getMap("N", NSIDE=NSIDE_WORKING)
dlm = hp.map2alm(delta)
Cldd = hp.anafast(delta)

# "Convergence in the lens frame": like convergence but with different lensing kernel.
kappa_LF_map = getMap("../kap_lt4.5", NSIDE=NSIDE_WORKING) # Not entirely OK
Clpd = hp.anafast(kappa_LF_map, map2=delta)
Clpd /= ls*(ls+1.0) # "convergence" -> potential
Clpd[0] = 0 # zero mode

# Filtered temperature, density fields
almT_resc = hp.almxfl(almT, 1.0/ClTT)
almT_resc_l2 = hp.almxfl(almT, ls*(ls+1.0)/ClTT)
dlm_resc = hp.almxfl(dlm, Clpd/Cldd)
dlm_resc_l2 = hp.almxfl(dlm, ls*(ls+1.0)*Clpd/Cldd)
# Real-space maps of filtered fields
T_resc = hp.alm2map(almT_resc, NSIDE)
T_resc_l2 = hp.alm2map(almT_resc_l2, NSIDE)
d_resc = hp.alm2map(dlm_resc, NSIDE)
d_resc_l2 = hp.alm2map(dlm_resc_l2, NSIDE)

# Multiply real-space maps, obtain resulting alm's
interm_vlm = hp.map2alm( 0.5*T_resc*d_resc )
interm_vlm = hp.almxfl( interm_vlm, ls*(ls+1.0) ) # multiply previous line by l*(l+1)
interm_vlm += hp.map2alm( 0.5*T_resc*d_resc_l2 )
interm_vlm -= hp.map2alm( 0.5*T_resc_l2*d_resc )

N = 1.0

# Multiply real-space products by noise
v_reconstlm = hp.almxfl( interm_vlm, N )
v_reconst = hp.alm2map(v_reconstlm, NSIDE)
v_reconst_PS = hp.anafast(v_reconst)


M_map = getMap("M", NSIDE=NSIDE_WORKING)
p_theta_map = getMap("p_theta", NSIDE=NSIDE_WORKING)
v_theta_map = np.nan_to_num(p_theta_map/M_map)
p_phi_map = getMap("p_phi", NSIDE=NSIDE_WORKING)
v_phi_map = np.nan_to_num(p_phi_map/M_map)
# Angular gradient of transverse momentum field
dthetaVtheta = hp.alm2map_der1(hp.map2alm(v_theta_map), NSIDE_WORKING)[1]
dphiVphi = hp.alm2map_der1(hp.map2alm(v_phi_map), NSIDE_WORKING)[2]
dV = dthetaVtheta + dphiVphi # \grad*V = "dV"
# Inverse angular laplacian of transverse momentum ~ 1/grad * V
dVlm = hp.map2alm(dV)
lmax = hp.Alm.getlmax(dVlm.size)
ls = np.arange(lmax)
invlap = 1.0 / ls / (ls + 1.0)
invlap[0] = 0.0 # zero monopole (?)
dVlm = hp.almxfl(dVlm, invlap)
vt_map = hp.alm2map(dVlm, NSIDE_WORKING)



# Plot velocity power spectra
vt_PS = hp.anafast(vt_map)
psplot(vt_PS, label="True velocity", norm=True)
psplot(v_reconst_PS, label="Unnorm. reconstructed velocity", norm=True)
plt.legend()
plt.savefig(OUTPUT_DIR+"velocity_PS.png")
plt.close()

# Correlation between velocity maps
plt.semilogx( hp.anafast(vt_map, v_reconst)/np.sqrt(vt_PS*v_reconst_PS), label="vt_map")
plt.legend()
plt.savefig(OUTPUT_DIR+"corr_coeff.png")
plt.close()
