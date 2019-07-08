import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
from   cosmology import *

NSIDE = 4096

rho_m_0 = 2.775e11*omegam*h**2 # Msun/Mpc^3

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

try :
  print(reduced_catalog.shape)
except Exception as e :
  print("Reducing catalogue...")
  x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
  chi = np.sqrt(x**2+y**2+z**2)    # Mpc
  reduced_catalog = catalog[ (0<chi) & (chi<300) ]
  print("...done.")

x  = reduced_catalog[:,0];  y = reduced_catalog[:,1];  z = reduced_catalog[:,2] # Mpc (comoving)
R  = reduced_catalog[:,6] # Mpc
vx = reduced_catalog[:,3]; vy = reduced_catalog[:,4]; vz = reduced_catalog[:,5] # km/sec
pix = hp.vec2pix(NSIDE, x, y, z)

# convert to mass, comoving distance, radial velocity, redshfit, RA and DEc
M        = 4*np.pi/3.*rho_m_0*R**3    # Msun
chi      = np.sqrt(x**2+y**2+z**2)    # Mpc
vrad     = (x*vx + y*vy + z*vz) / chi # km/sec
redshift = zofchi(chi)

theta, phi  = hp.vec2ang(np.column_stack((x,y,z))) # in radians

plt.hist(redshift, bins=100)
plt.savefig('redshift_histogram.png')
plt.close()

plt.hist(chi, bins=100)
plt.savefig('chi_histogram.png')
plt.close()

### project to a map, matching the websky orientations
def mapof(name, weights, title=None) :
  _map = np.zeros((hp.nside2npix(NSIDE)), dtype=weights.dtype)
  print("Generating",name,"map of type",_map.dtype)
  np.add.at(_map, pix, weights)
  hp.mollview(_map)
  if title is not None :
    plt.title(title)
  else :
    plt.title(name + ' map')
  plt.savefig(name+'_map.png')
  plt.close()
  hp.fitsfunc.write_map(name+'.fits', _map, overwrite=True)

mapof("N", (0.0*M+1.0).astype(np.int) )
mapof("M", M)
mapof("rho", M*(1.0 + redshift)**3) # integrated density
mapof("prad", vrad*M*(1.0 + redshift)**3)
mapof("tau", M*(1.0 + redshift)**2) # integrated tau
