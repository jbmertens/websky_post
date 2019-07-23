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
NSIDE_MAX = 4096
pywig.wig_table_init(3*NSIDE_MAX, 3)
pywig.wig_temp_init(3*NSIDE_MAX)


parser = argparse.ArgumentParser(description='Scripts to perform reconstruction from websky data.')

parser.add_argument('-vc', '--velocity_comparison', help='Compare velocity maps generated using different methods.', action='store_true')
parser.add_argument('-kc', '--ksz_comparison', help='Compare halo and websky maps.', action='store_true')
parser.add_argument('-kr', '--ksz_reconstruction', help='Perform kSZ reconstruction.', action='store_true')

if not len(sys.argv) > 1 :
  parser.print_help()

args = parser.parse_args()

#################
## Import maps ##
#################

try :
  len(LOADED_FITS)
except Exception as e:
  LOADED_FITS = {}
def getFits(filepath, reload=False) :
  if filepath in LOADED_FITS and not reload:
    data = LOADED_FITS[filepath]
  else :
    data = hp.fitsfunc.read_map(filepath)
    LOADED_FITS[filepath] = data
  return data

def getMap(name, NSIDE=None) :
  _map = getFits(MAPS_OUTPUT_DIR+name+".fits")

  if NSIDE is not None :
    return hp.ud_grade(_map, NSIDE)
  else :
    return _map

# Coarsened/interpolated velocity maps
def getVelocity() :
  return 0


#############################
## Velocity map comparison ##
#############################

if args.velocity_comparison :
  mkdir_p(MAPS_OUTPUT_DIR+"vc")
  NSIDE_COARSE = 512

  vrad = getMap("vrad", NSIDE=NSIDE_COARSE)
  print("( Mean , std ) vrad = (", np.mean(vrad), ",", np.std(vrad), ")" )

  prad = getMap("prad", NSIDE=NSIDE_COARSE)
  M = getMap("M", NSIDE=NSIDE_COARSE)
  vrad_pM = prad/M

  print("( Mean , std ) vrad_pM = (", np.mean(vrad_pM), ",", np.std(vrad_pM), ")" )

  plt_min, plt_max = np.percentile(vrad, (1.0, 99.0))
  hp.mollview(vrad, min=plt_min, max=plt_max)
  plt.title('Direct, Averaged velocity map')
  plt.savefig(MAPS_OUTPUT_DIR+'vc/vcoarse_map.png')
  plt.close()

  plt_min, plt_max = np.percentile(vrad_pM, (1.0, 99.0))
  hp.mollview(vrad_pM, min=plt_min, max=plt_max)
  plt.title('Momentum/Mass velocity map')
  plt.savefig(MAPS_OUTPUT_DIR+'vc/vrad_map.png')
  plt.close()

  plt.loglog(hp.anafast(vrad), label="vrad")
  plt.loglog(hp.anafast(vrad_pM), label="vrad_pM")
  plt.legend()
  plt.savefig(MAPS_OUTPUT_DIR+"vc/sim_velocity_comp_power_spectra.png")
  plt.close()

  plt.semilogx(hp.anafast(vrad,vrad_pM)/np.sqrt(hp.anafast(vrad)*hp.anafast(vrad_pM)))
  plt.savefig(MAPS_OUTPUT_DIR+"vc/sim_velocity_comp_corr_coeff.png")
  plt.close()


#################################
## kSZ dipole field comparison ##
#################################

if args.ksz_comparison :
  mkdir_p(MAPS_OUTPUT_DIR+"kc")
  NSIDE_COARSE = 64

  ksz_websky = getMap("../ksz", NSIDE=NSIDE_COARSE)
  print("( Mean , std ) ksz_websky = (", np.mean(ksz_websky), ",", np.std(ksz_websky), ")" )
  ksz_halo = getMap("ksz", NSIDE=NSIDE_COARSE)
  print("( Mean , std ) ksz_halo = (", np.mean(ksz_halo), ",", np.std(ksz_halo), ")" )

  plt_min, plt_max = np.percentile(ksz_websky, (0.1, 99.9))
  hp.mollview(ksz_websky, min=plt_min, max=plt_max)
  plt.title('Websky kSZ map')
  plt.savefig(MAPS_OUTPUT_DIR+'kc/ksz_websky.png')
  plt.close()

  plt_min, plt_max = np.percentile(ksz_halo, (0.1, 99.9))
  hp.mollview(ksz_halo, min=plt_min, max=plt_max)
  plt.title('Halo catalogue kSZ map')
  plt.savefig(MAPS_OUTPUT_DIR+'kc/ksz_halo.png')
  plt.close()

  plt.loglog(hp.anafast(ksz_websky), label="ksz_websky")
  plt.loglog(hp.anafast(ksz_halo), label="ksz_halo")
  plt.legend()
  plt.savefig(MAPS_OUTPUT_DIR+"kc/ksz_sim_comparison_powerspec.png")
  plt.close()

  plt.semilogx(hp.anafast(ksz_halo,ksz_websky)/np.sqrt(hp.anafast(ksz_halo)*hp.anafast(ksz_websky)))
  plt.savefig(MAPS_OUTPUT_DIR+"kc/ksz_sim_comparison_corr_coeff.png")
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

if args.ksz_reconstruction :
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
