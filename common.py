from scipy.interpolate import interp1d
import numpy as np
import healpy as hp
import errno, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

omegab = 0.049
omegac = 0.261
omegam = omegab + omegac
h      = 0.68
ns     = 0.965
sigma8 = 0.81

c = 3e5

H0 = 100*h

nz = 100000
z1 = 0.0
z2 = 6.0
za = np.linspace(z1,z2,nz)
dz = za[1]-za[0]

Hofz   = lambda z: H0*np.sqrt(omegam*(1+z)**3+1-omegam)
dchidz = lambda z: c/Hofz(z)

chia = np.cumsum(dchidz(za))*dz

zofchi = interp1d(chia,za)
chiofz = interp1d(za,chia)

rho_m_0 = 2.775e11*omegam*h**2 # Msun/Mpc^3

sigma_T = 6.9868448e-74 # Thompson cross section in Mpc^2
# parameters consistent with https://arxiv.org/pdf/1511.02843.pdf
YHe = 0.2477 # Helium fraction
mHe = 3.343e-57 # Msun
mH = 8.42e-58 # Msun
mu = (1.0-YHe)/mH + YHe/mHe # 1/Msun
fb = omegab/omegam
ksz_prefac = -sigma_T * fb*mu # * T_CMB


NSIDE = 4096
NPIX = 12*NSIDE**2
OMEGA_PIX = 4.0*np.pi/NPIX

Z_MIN = 0.40
Z_MAX = 0.50

CHI_MIN = chiofz(Z_MIN)
CHI_MAX = chiofz(Z_MAX)

M_CUT_PERCENTILE = 0.0 # 99.999 is top 0.001% of halos

#################
## Working dir ##
#################

MAPS_OUTPUT_DIR = "halomaps.z"+("%.2f" % Z_MIN)+"-z"+("%.2f" % Z_MAX)+".mp"+("%.4f" % M_CUT_PERCENTILE)+"_NSIDE-"+str(NSIDE)+"/"

def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

mkdir_p(MAPS_OUTPUT_DIR)

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

def nsideof(_map) :
  return int(round(np.sqrt(len(_map)//12)))

def getMap(name, NSIDE=None) :
  _map = getFits(MAPS_OUTPUT_DIR+name+".fits")

  if (NSIDE is not None) and (NSIDE != nsideof(_map)) :
    return hp.ud_grade(_map, NSIDE)
  else :
    return _map

def psplot(ps, label=None, norm=False) :
  ls = np.arange(len(ps))[1:]
  if norm :
    mean_ps = np.mean(ps[1:])
    plt.loglog(ls, ls*(ls+1.0)*ps[1:]/mean_ps, label=label)
  else :
    plt.loglog(ls, ls*(ls+1.0)*ps[1:], label=label)
