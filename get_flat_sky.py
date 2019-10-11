import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import sys, argparse, multiprocessing
from common import *

parser = argparse.ArgumentParser(description='Scripts to generate "flat" sky patches.')

parser.add_argument('-train', '--training_patch', help='Generate sky patches in training region.', action='store_true')
parser.add_argument('-test', '--testing_patch', help='Generate sky patches in testing region.', action='store_true')
parser.add_argument('-n', '--number_patches', help='Number of patches to generate. (Default 10)', action='store', type=int, default=10)
parser.add_argument('-r', '--resolution', help='Resolution of patch (r by r array, default r=128).', action='store', type=int, default=128)
parser.add_argument('-s', '--size', help='Angular width of patch (s by s, with s in degrees, default s=5).', action='store', type=float, default=5)
if not len(sys.argv) > 1 :
  parser.print_help()
args = parser.parse_args()

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def store_patch(direction, map_data, filename) :
  """
  Save a "patch" of sky to a npy file.
  """

  # begin with "patch" in (1,0,0)-direction
  PATCH_DX = args.size/180.0 * 1.0/args.resolution
  ord_pixel_vecs = np.array([
    [1.0, y*PATCH_DX, z*PATCH_DX]
      for y in np.arange(-args.resolution//2, args.resolution//2)
      for z in np.arange(-args.resolution//2, args.resolution//2) ])

  # rotate patch to "correct" direction.
  # Cross product of vectors provides a direction to rotate about,
  # dot product of vectors provides angle to rotate by.
  dhat = np.array(direction)/np.sqrt(np.dot(direction, direction)) # normalized direction
  xhat = np.array([1.0, 0.0, 0.0])
  axis = np.cross(xhat, dhat)
  theta = np.arccos(np.dot(xhat, dhat))
  rot_mat = rotation_matrix(axis, theta)
  patch_pixel_dirs = np.array([hp.vec2ang(np.matmul(rot_mat, vec)) for vec in ord_pixel_vecs])

  # get map values in pixels and save
  np.save(filename,
    hp.get_interp_val(map_data, patch_pixel_dirs[:,0], patch_pixel_dirs[:,1]) )



# ##
# Actual logic below...
# ##

if args.training_patch or args.testing_patch:
  # consider x>0,y>0,z>0 to be "test" data, get training data outside this area
  TRAINING_OUTPUT_DIR = MAPS_OUTPUT_DIR+"training/"
  TESTING_OUTPUT_DIR = MAPS_OUTPUT_DIR+"testing/"
  mkdir_p(TRAINING_OUTPUT_DIR)
  mkdir_p(TESTING_OUTPUT_DIR)
  NSIDE_WORKING = NSIDE

  rho_map = getMap("N_uncertain", NSIDE=NSIDE_WORKING) # _uncertain
  
  vrad_map = getMap("vrad", NSIDE=NSIDE_WORKING) # Directly averaged velocity
  ksz_map = getMap("../ksz", NSIDE=NSIDE_WORKING) # websky kSZ
  try:
    _CMB_foo = CMB_map[0]
  except Exception as e:
    CMB_alms = hp.fitsfunc.read_alm('lensed_alm.fits').astype(np.complex)
    CMB_map = hp.alm2map(CMB_alms, NSIDE_WORKING)
  patchy_ksz_map = getMap("../ksz_patchy", NSIDE=NSIDE_WORKING) # websky patchy kSZ
  tsz_map = T_CMB*y_to_tSZ*getMap("../tsz", NSIDE=NSIDE_WORKING) # websky tSZ
  Obs_T_map = ksz_map + CMB_map + patchy_ksz_map + tsz_map


  n_training_patches = 0
  n_testing_patches = 0
  while n_training_patches < args.number_patches and n_testing_patches < args.number_patches:
    x = np.random.random()*2-1
    y = np.random.random()*2-1
    z = np.random.random()*2-1

    if x**2 + y**2 + z**2 < 1 : # inside sphere (randomly distributed direction)
      out_dir = 0
      if (x<0 or y<0 or z<0) : # outisde (+,+,+)-quadrant = training patches
        if n_training_patches < args.number_patches :
          out_dir = TRAINING_OUTPUT_DIR+str(n_training_patches)+"-"
          n_training_patches += 1
      else :
        if n_testing_patches < args.number_patches :
          out_dir = TESTING_OUTPUT_DIR+str(n_testing_patches)+"-"
          n_testing_patches += 1

      if out_dir :
        store_patch([x,y,z], vrad_map, out_dir+"vrad-patch")
        store_patch([x,y,z], rho_map, out_dir+"rho-patch")
        store_patch([x,y,z], Obs_T_map, out_dir+"Obs_T-patch")
