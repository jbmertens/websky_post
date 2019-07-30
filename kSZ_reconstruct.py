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


#############################
## Velocity map comparison ##
#############################

if args.velocity_comparison :
  """Compare power spectra of different velocity maps"""

  VC_OUTPUT_DIR = MAPS_OUTPUT_DIR+"vc/"
  mkdir_p(VC_OUTPUT_DIR)
  NSIDE_COARSE = 256

  vrad = getMap("vrad")
  vrad_coarse = getMap("vrad", NSIDE=NSIDE_COARSE)
  vrad_conf = getMap("vrad_conf")
  vrad_conf_coarse = getMap("vrad_conf", NSIDE=NSIDE_COARSE)

  prad = getMap("prad")
  tau = getMap("tau")
  vrad_ptau = np.nan_to_num(prad/tau)

  prad_coarse = getMap("prad", NSIDE=NSIDE_COARSE)
  tau_coarse = getMap("tau", NSIDE=NSIDE_COARSE)
  vrad_ptau_coarse = np.nan_to_num(prad_coarse/tau_coarse)

  vrad_coarse_2 = np.nan_to_num(hp.ud_grade(vrad*tau, NSIDE_COARSE)/tau_coarse)

  psplot(hp.anafast(vrad), label="vrad", norm=True)
  psplot(hp.anafast(vrad_coarse), label="vrad_coarse", norm=True)
  psplot(hp.anafast(vrad_conf), label="vrad_conf", norm=True)
  psplot(hp.anafast(vrad_conf_coarse), label="vrad_conf_coarse", norm=True)
  psplot(hp.anafast(vrad_coarse_2), label="vrad_coarse_2", norm=True)
  psplot(hp.anafast(vrad_ptau), label="vrad_ptau", norm=True)
  psplot(hp.anafast(vrad_ptau_coarse), label="vrad_ptau_coarse", norm=True)
  # psplot(unnorm_veff_reconst_ps_scaled, label="unnorm_veff_reconst", norm=True)
  plt.legend()
  plt.savefig(VC_OUTPUT_DIR+"sim_velocity_comp_power_spectra.png")
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


####################
## Reconstruction ##
####################

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

if args.ksz_reconstruction :

  NSIDE_WORKING = NSIDE
  OUTPUT_DIR = MAPS_OUTPUT_DIR+"ksz_reconstruction/"
  mkdir_p(OUTPUT_DIR)

  tau_map = getMap("tau", NSIDE=NSIDE_WORKING)
  rho_map = getMap("N", NSIDE=NSIDE_WORKING)
  vrad_map = getMap("vrad", NSIDE=NSIDE_WORKING) # Directly averaged velocity

  ksz_map = getMap("../ksz", NSIDE=NSIDE_WORKING) # websky kSZ
  # ksz_map = getMap("ksz", NSIDE=NSIDE_WORKING) # kSZ for this bin only
  # ksz_map = getMap("../ksz_halos", NSIDE=NSIDE_WORKING) # kSZ from halo catalogue only

  CMB_alms = hp.fitsfunc.read_alm('lensed_alm.fits').astype(np.complex)
  CMB_map = hp.alm2map(CMB_alms, NSIDE_WORKING)

  Obs_T_map = ksz_map + CMB_map

  # CMB power spectra
  CMB_PS = hp.anafast(CMB_map)
  ksz_PS = hp.anafast(ksz_map)
  Obs_T_PS = hp.anafast(Obs_T_map)
  psplot(CMB_PS, label="CMB")
  psplot(ksz_PS, label="kSZ")
  psplot(Obs_T_PS, label="Total")
  plt.legend()
  plt.savefig(OUTPUT_DIR+"CMB_PS.png")
  plt.close()


  # try reconstructing...
  print("Generating power spectra.")
  ClTT = hp.anafast(Obs_T_map)
  Cldd = hp.anafast(rho_map)
  Cltd = hp.anafast(rho_map, map2=tau_map)
  ls = np.arange(ClTT.size)

  print("Generating alms.")
  dTlm = hp.map2alm(Obs_T_map)
  dlm = hp.map2alm(rho_map)
  CldTdT = ClTT

  print("Generating rescaled alms.")
  dTlm_resc = hp.almxfl(dTlm, 1.0/CldTdT)
  dT_resc = hp.alm2map(dTlm_resc, NSIDE)
  dlm_resc = hp.almxfl(dlm, Cltd/Cldd)
  d_resc = hp.alm2map(dlm_resc, NSIDE)


  # Compute noise (expensive, need to optimize?)
  # print("Computing noise.")
  # ncores = multiprocessing.cpu_count()
  # Ninv = [ getNinv(l, ls, Cltd, CldTdT, Cldd) for l in ls ]
  # Ninv = Parallel(n_jobs=ncores)(delayed(getNinv)(l, ls, Cltd, CldTdT, Cldd) for l in ls)
  # N = 1.0/np.array(Ninv)
  # N = np.zeros_like(ls, dtype=np.int)
  # N[:100] = 1.0

  unnorm_veff_reconstlm = hp.map2alm(dT_resc*d_resc)
  unnorm_veff_reconst_ps = hp.alm2cl(unnorm_veff_reconstlm)

  N = np.sqrt( hp.anafast(vrad_map) / unnorm_veff_reconst_ps ) # Fake noise just to make the reconstructed spectrum agree.
  veff_reconstlm = hp.almxfl(unnorm_veff_reconstlm, N)
  veff_reconst = hp.alm2map(veff_reconstlm, NSIDE)

  dT_resc_ps = hp.anafast(dT_resc)
  d_resc_ps = hp.anafast(d_resc)
  psplot(dT_resc_ps)
  psplot(d_resc_ps)
  plt.savefig(OUTPUT_DIR+'interm_ps.png')
  plt.close()

  # # Plot reconstructed velocity maps
  # hp.mollview(veff_reconst)
  # plt.title('Reconstructed velocity map')
  # plt.savefig(OUTPUT_DIR+'veff_reconst_map.png')
  # plt.close()
  # hp.mollview(vrad_map)
  # plt.title('Simulated velocity map')
  # plt.savefig(OUTPUT_DIR+'veff_map.png')
  # plt.close()

  # Plot velocity power spectra
  vrad_PS = hp.anafast(vrad_map)
  psplot(vrad_PS, label="True velocity", norm=True)
  psplot(unnorm_veff_reconst_ps, label="Unnorm. reconstructed velocity", norm=True)
  plt.legend()
  plt.savefig(OUTPUT_DIR+"velocity_PS.png")
  plt.close()

  # Correlation between velocity maps
  plt.semilogx( hp.anafast(vrad_map, veff_reconst)/np.sqrt(hp.anafast(vrad_map)*hp.anafast(veff_reconst)), label="vrad_map")
  plt.legend()
  plt.savefig(OUTPUT_DIR+"corr_coeff.png")
  plt.close()
