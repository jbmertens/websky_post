import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
from common import *

f=open('halos.pksc')
# only take first five entries for testing (there are ~8e8 halos total...)
# N = 5
# uncomment the following line to read in all halos
N=np.fromfile(f,count=3,dtype=np.int32)[0]

try :
  print(catalog.shape)
except Exception as e :
  catalog=np.fromfile(f,count=N*10,dtype=np.float32)
  catalog=np.reshape(catalog,(N,10))

print("Reducing catalogue...")
x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
chi = np.sqrt(x**2+y**2+z**2)    # Mpc
reduced_catalog = catalog[ (chi>CHI_MIN) & (chi<CHI_MAX) ]
print("...done.")

x  = reduced_catalog[:,0];  y = reduced_catalog[:,1];  z = reduced_catalog[:,2] # Mpc (comoving)
R  = reduced_catalog[:,6] # Mpc
vx = reduced_catalog[:,3]; vy = reduced_catalog[:,4]; vz = reduced_catalog[:,5] # km/sec
pix = hp.vec2pix(NSIDE, x, y, z)

# convert to mass, comoving distance, radial velocity, redshift, RA and DEc
M        = 4*np.pi/3.*rho_m_0*R**3    # Msun
chi      = np.sqrt(x**2+y**2+z**2)    # Mpc
vrad     = (x*vx + y*vy + z*vz) / chi # km/sec
redshift = zofchi(chi)
dA       = chi / (1.0 + redshift)     # Mpc (angular diameter distance)

theta, phi  = hp.vec2ang(np.column_stack((x,y,z))) # in radians

plt.hist(redshift, bins=100)
plt.savefig(MAPS_OUTPUT_DIR+'redshift_histogram.png')
plt.close()

plt.hist(chi, bins=100)
plt.savefig(MAPS_OUTPUT_DIR+'chi_histogram.png')
plt.close()


def nearest_neighbor_fill(_map, nside_coarse, weights=1.0) :
  # Coarsen map by filling in missing data with lower nside values
  # Fill in by averaging over nonzero values in cell:
  #   coarse = <fine*weights>/<weights>
  tol = 1.0e-9

  nonzero_cells_mask = weights*( np.abs(_map) > tol )
  map_dgraded = hp.ud_grade(weights*_map, nside_out=nside_coarse)
  nonzero_cells_mask_dgraded = hp.ud_grade(nonzero_cells_mask, nside_out=nside_coarse)
  map_replacements = hp.ud_grade( np.nan_to_num( map_dgraded / nonzero_cells_mask_dgraded ), nside )

  zero_cells = np.where(np.abs(_map) <= tol)
  _map[zero_cells] = map_replacements[zero_cells]
  return _map

def nearest_neighbor_fill_iterate(_map, nside_coarse, weights=1.0) :
  working_nside = nsideof(_map)
  while working_nside >= nside_coarse :
    working_nside = working_nside//2
    _map = nearest_neighbor_fill(_map, working_nside, weights=1.0)
  return _map

def mapof(name, weights, title=None, fill_weights=None) :
  
  _map = np.zeros((hp.nside2npix(NSIDE)), dtype=weights.dtype)
  print("Generating",name,"map of type",_map.dtype)

  np.add.at(_map, pix, weights)
  if fill_weights is not None :
    _map = nearest_neighbor_fill_iterate(_map, 128, fill_weights)
  
  hp.mollview(_map)
  plt.title(name + ' map')
  plt.savefig(MAPS_OUTPUT_DIR+name+'_map.png')
  plt.close()
  
  hp.fitsfunc.write_map(MAPS_OUTPUT_DIR+name+'.fits', _map, overwrite=True)


mapof("N", (0.0*M+1.0).astype(np.int) ) # number counts map

mapof("vrad", vrad) # averaged velocity

mapof("prad", M*vrad/dA/dA/OMEGA_PIX ) # radial momentum
mapof("ksz", ksz_prefac*M*vrad/dA/dA/OMEGA_PIX ) # kSZ temperature

mapof("tau", ksz_prefac*M/dA/dA/OMEGA_PIX) # Optical depth (~ kSZ without velocity)


# Additional possibly useful maps
# mapof("M", M) # Mass map
# mapof("rho", M / (1.0+redshift)**3 / dA**2 / OMEGA_PIX/rho_m_0 ) # proper-distance-weighted density integral
# mapof("rho_conf", M / (1.0+redshift)**4 / dA**2 / OMEGA_PIX/rho_m_0 ) # conformal-distance-weighted density integral
# mapof("vrad_M", M*vrad) # Mass-weighted velocity
# mapof("vrad_fill", vrad, 1.0) # averaged velocity
# mapof("vrad_fill_M", vrad, M) # averaged velocity
# mapof("vrad_conf", vrad / (1.0+redshift)) # conformal-distance-weighted average velocity
