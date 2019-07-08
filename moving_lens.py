import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
from cosmology import *

rho_m_0 = 2.775e11*omegam*h**2 # Msun/Mpc^3

NSIDE = 2048
NPIX = 12*NSIDE**2

print("Computing moving lens temperature perturbations.")

# Load in catalog
f=open('halos.pksc')
N=np.fromfile(f,count=3,dtype=np.int32)[0]
catalog=np.fromfile(f,count=N*10,dtype=np.float32)
catalog=np.reshape(catalog,(N,10))
x = catalog[:,0]; y = catalog[:,1]; z = catalog[:,2] # Mpc (comoving)
chi = np.sqrt(x**2 + y**2 + z**2) # Mpc

# Bins to compute transverse perturbations in
chi_bins = np.linspace(0, 5000, 200)

dT_ML_lms = hp.map2alm(np.zeros((hp.nside2npix(NSIDE)), dtype=np.float))

for bin_num in range(len(chi_bins)-1) :
  print("Working in bin", bin_num, "of", len(chi_bins), "...")

  print("Reducing catalogue...")
  bin_lower_chi = chi_bins[bin_num]
  bin_upper_chi = chi_bins[bin_num+1]
  d_chi = bin_upper_chi - bin_lower_chi
  reduced_catalog = catalog[ (bin_lower_chi<chi) & (chi<bin_upper_chi) ]
  print("...done.")

  x_red = reduced_catalog[:,0]
  y_red = reduced_catalog[:,1]
  z_red = reduced_catalog[:,2]

  vx = reduced_catalog[:,3] / c
  vy = reduced_catalog[:,4] / c
  vz = reduced_catalog[:,5] / c # km/sec

  R = reduced_catalog[:,6] # Halo radius, Mpc
  M = 4*np.pi/3.*rho_m_0*R**3 # Halo masses, msun
  
  pix = hp.vec2pix(NSIDE, x_red, y_red, z_red)

  # Transverse momentum field
  # for each halo in chi-bin:
  theta, phi  = hp.vec2ang(np.column_stack((x_red, y_red, z_red))) # in radians
  p_theta = M * ( np.cos(theta)*np.cos(phi)*vx + np.cos(theta)*np.sin(phi)*vy - np.sin(theta)*vz )
  p_phi = M * ( -np.sin(theta)*vx + np.cos(theta)*vy )
  # Map of transverse momentum 
  p_theta_map = np.zeros((hp.nside2npix(NSIDE)), dtype=np.float)
  np.add.at(p_theta_map, pix, p_theta)
  p_phi_map = np.zeros((hp.nside2npix(NSIDE)), dtype=np.float)
  np.add.at(p_phi_map, pix, p_phi)

  # Angular gradient of transverse momentum field
  dthetaPtheta = hp.alm2map_der1(hp.map2alm(p_theta_map), NSIDE)[1]
  dphiPphi = hp.alm2map_der1(hp.map2alm(p_phi_map), NSIDE)[2]
  dP = dthetaPtheta + dphiPphi # \grad*P = "dP"
  # Inverse angular laplacian of transverse momentum ~ 1/grad * P
  dPlm = hp.map2alm(dP)
  lmax = hp.Alm.getlmax(dPlm.size)
  ls = np.arange(lmax)
  invlap = 1.0 / ls / (ls + 1.0)
  invlap[0] = 1.0 # preserve monopole (?)
  dPlm = hp.almxfl(dPlm, invlap)
  
  # contribution to map from each radial bin
  # Contribution is -2 \int dr d/dt \Psi = -2 \int dr -3/2 H^2 1/\grad P
  # P is (1+\delta)v = M/<M> v; <M> = M_pix is avg M per pixel in shell
  M_pix = 4.0/3.0*np.pi*( bin_upper_chi**3 - bin_lower_chi**3 )*rho_m_0 / NPIX
  print("Computing dT_in_bin.")
  # total map is this integrated in all bins * geometric factors...
  chi_bin = (bin_lower_chi+bin_upper_chi)/2.0
  z_bin = zofchi(chi_bin)
  a_bin = 1.0 / (1.0 + z_bin)
  H_bin = H(z_bin)/c
  dT_ML_lms += 3.0 * d_chi * (a_bin*H_bin)**2 * chi_bin * dPlm / M_pix

# Store & Plot result.
dT_ML = hp.alm2map(dT_ML_lms, NSIDE)
hp.mollview(dT_ML)
plt.title('Moving Lens sky')
plt.savefig('Tmap_ML.png')
plt.close()
hp.fitsfunc.write_map('ML.fits', dT_ML, overwrite=True)
