import h5py
import scipy.io

dir = '/home/wagenaar/python/daw/test'
#f = h5py.File(f'{dir}/test-hdf.mat', 'r')

f = scipy.io.loadmat(f'{dir}/test-v6.mat')

f = scipy.io.loadmat(f'{dir}/test-octave-binary.mat')

