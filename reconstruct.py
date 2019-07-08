import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import pywigxjpf as pywig
from joblib import Parallel, delayed
import multiprocessing

NSIDE = 4096
print("  Creating pywig tables...")
pywig.wig_table_init(3*NSIDE, 3)
pywig.wig_temp_init(3*NSIDE)


#################
## Import maps ##
#################

# Radial momentum map
try:
  val = np.min(prad_map); print("prad.fits loaded.")  
except Exception as e:
  prad_map = hp.fitsfunc.read_map('prad.fits')
  prad_map = hp.ud_grade(prad_map, NSIDE)

# load kSZ map
try:
  val = np.min(ksz_map); print("ksz.fits loaded.")  
except Exception as e:
  ksz_map = hp.fitsfunc.read_map('ksz.fits')
  ksz_map = hp.ud_grade(ksz_map, NSIDE)

# load density map
try:
  val = np.min(M_map); print("N.fits loaded.")  
except Exception as e:
  M_map = hp.fitsfunc.read_map('N.fits').astype(np.float)
  M_map = hp.ud_grade(M_map, NSIDE)

# load tau map
try:
  val = np.min(tau_map); print("tau.fits loaded.")  
except Exception as e:
  tau_map = hp.fitsfunc.read_map('tau.fits').astype(np.float)
  tau_map = hp.ud_grade(tau_map, NSIDE)

# load (lensed) CMB map
try:
  val = np.min(CMB_map); print("CMB.fits loaded.")  
except Exception as e:
  CMB_alms = hp.fitsfunc.read_alm('lensed_alm.fits').astype(np.complex)
  CMB_map = hp.alm2map(CMB_alms, NSIDE)


# velocity maps
def getVelocity() :
  return 0

NSIDE_COARSE=128
vrad_map = np.nan_to_num(prad_map/M_map)
vrad_coarse = np.nan_to_num(hp.ud_grade(prad_map, NSIDE_COARSE)/hp.ud_grade(M_map, NSIDE_COARSE))
veff_mean = np.mean(vrad_coarse)
veff_std = np.std(vrad_coarse)
plt_min = veff_mean - 3.0*veff_std
plt_max = veff_mean + 3.0*veff_std
hp.mollview(vrad_coarse, min=plt_min, max=plt_max)
plt.title('Coarsened velocity map')
plt.savefig('vcoarse_map.png')
plt.close()
hp.mollview(vrad_map, min=plt_min, max=plt_max)
plt.title('Velocity map')
plt.savefig('vrad_map.png')
plt.close()


##########################
## Reconstruction Noise ##
##########################

L_RECONST_MAX = 50

def GammakSZ(l1, l2, l, Cltd) :
  pref = np.sqrt((2.0*l1+1)*(2.0*l2+1)*(2.0*l+1)/4.0/np.pi)
  wig = pywig.wig3jj(2*l1, 2*l2, 2*l, 0, 0, 0) # 2*[j1,j2,j3,m1,m2,m3]
  return pref*wig*Cltd[l2]

def getNinv(l, ls, Cltd, ClTT, Cldd) :
  Ninv = 0.0
  if l < L_RECONST_MAX+1 :
    print("Working on l =", l)
    for l1 in ls: # TODO: don't include monopole, dipole contributions?
      for l2 in ls:
        Ninv += GammakSZ(l1, l2, l, Cltd)**2 / ClTT[l1] / Cldd[l2]
    Ninv /= (2.0*l + 1.0)
  else :
    Ninv = 1.0e50 # N = 1.0e-50
  return Ninv

def psplot(ps, label=None) :
  ls = np.arange(len(ps))[2:]
  plt.loglog(ls, ls*(ls+1.0)*ps[2:], label=label)


####################
## Reconstruction ##
####################

Obs_T_map = ksz_map + CMB_map

# CMB power spectra
CMB_PS = hp.anafast(CMB_map)
ksz_PS = hp.anafast(ksz_map)
Obs_T_PS = hp.anafast(Obs_T_map)
psplot(CMB_PS)
psplot(ksz_PS)
psplot(Obs_T_PS)
plt.savefig("CMB_PS.png")
plt.close()


# try reconstructing...
print("Generating power spectra.")
ClTT = hp.anafast(Obs_T_map)
Cldd = hp.anafast(M_map)
Cltd = hp.anafast(M_map, map2=tau_map)
ls = np.arange(ClTT.size)

print("Generating alms.")
dTlm = hp.map2alm(Obs_T_map)
dlm = hp.map2alm(M_map)
CldTdT = ClTT

print("Generating rescaled alms.")
dTlm_resc = hp.almxfl(dTlm, 1.0/CldTdT)
dT_resc = hp.alm2map(dTlm_resc, NSIDE)
dlm_resc = hp.almxfl(dlm, Cltd/Cldd)
d_resc = hp.alm2map(dlm_resc, NSIDE)


# print("Computing noise.")
# ncores = multiprocessing.cpu_count()
# Ninv = [ getNinv(l, ls, Cltd, CldTdT, Cldd) for l in ls ]
# Ninv = Parallel(n_jobs=ncores)(delayed(getNinv)(l, ls, Cltd, CldTdT, Cldd) for l in ls)
# N = 1.0/np.array(Ninv)
# N = np.zeros_like(ls, dtype=np.int)
# N[:100] = 1.0

# Try multiplying maps at higher NSIDE
# unnorm_veff_udgraded = hp.ud_grade( hp.ud_grade(dT_resc, 2*NSIDE)*hp.ud_grade(d_resc, 2*NSIDE), NSIDE )
# unnorm_veff_udgraded_reconstlm = hp.map2alm(unnorm_veff_udgraded)
# unnorm_veff_udgraded_reconst_ps = hp.alm2cl(unnorm_veff_udgraded_reconstlm)
# N = np.sqrt( hp.anafast(vrad_map) / unnorm_veff_udgraded_reconst_ps )
# veff_reconstlm = hp.almxfl(unnorm_veff_udgraded_reconstlm, N)
# veff_reconst = hp.alm2map(veff_reconstlm, NSIDE)

def nsideOf(map):
  return np.sqrt(map.shape[0]/12)

unnorm_veff_reconstlm = hp.map2alm(dT_resc*d_resc)
unnorm_veff_reconst_ps = hp.alm2cl(unnorm_veff_reconstlm)
N = np.sqrt( hp.anafast(vrad_map) / unnorm_veff_reconst_ps )
veff_reconstlm = hp.almxfl(unnorm_veff_reconstlm, N)
veff_reconst = hp.alm2map(veff_reconstlm, NSIDE)


dT_resc_ps = hp.anafast(dT_resc)
d_resc_ps = hp.anafast(d_resc)
psplot(dT_resc_ps)
psplot(d_resc_ps)
try:
  psplot(unnorm_veff_reconst_ps)
except Exception as e:
  pass
try:
  psplot(unnorm_veff_udgraded_reconst_ps)
except Exception as e:
  pass
plt.savefig('interm_ps.png')
plt.close()



# Plot reconstructed velocity
veff_reconst_coarse = hp.ud_grade(veff_reconst, NSIDE_COARSE)
veff_mean = np.mean(veff_reconst_coarse)
veff_std = np.std(veff_reconst_coarse)
plt_min = veff_mean - 3.0*veff_std
plt_max = veff_mean + 3.0*veff_std
hp.mollview(-veff_reconst_coarse, min=plt_min, max=plt_max)
plt.title('Reconstructed velocity map')
plt.savefig('veff_reconst_map.png')
plt.close()

# Correlation with velocity map
rl = hp.anafast(vrad_coarse,-veff_reconst_coarse)/np.sqrt(hp.anafast(vrad_coarse)*hp.anafast(veff_reconst_coarse))
plt.semilogx(rl, label="Coarse correlation")
rl = hp.anafast(vrad_map,-veff_reconst)/np.sqrt(hp.anafast(vrad_map)*hp.anafast(veff_reconst))
plt.semilogx(rl, label="plain correlation")
plt.legend()
plt.savefig("corr_coeff.png")
plt.close()

# Plot different velocity power spectra
vrad_PS = hp.anafast(vrad_map)
vrad_coarse_PS = hp.anafast(vrad_coarse)
veff_reconst_coarse_PS = hp.anafast(veff_reconst_coarse)

psplot(vrad_PS*vrad_PS[1]/vrad_PS[1], label="True velocity")
psplot(vrad_coarse_PS*vrad_PS[1]/vrad_coarse_PS[1], label="Coarsened true velocity")
psplot(veff_reconst_coarse_PS*vrad_PS[1]/veff_reconst_coarse_PS[1], label="Reconstructed velocity")
plt.legend()
plt.savefig("velocity_PS.png")
plt.close()
