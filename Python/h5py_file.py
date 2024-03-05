import numpy as np
import h5py

matrix_1 = np.random.random(size=(1000,1000))
matrix_2 = np.random.random(size=(1000,1000))
matrix_3 = np.random.random(size=(1000,1000))
matrix_4 = np.random.random(size=(1000,1000))

with h5py.File('./hdf5_groups.h5', 'w') as hdf:
    # Create group
    G1 = hdf.create_group('/Group1')
    G1.create_dataset('dataset1', data=matrix_1)
    G1.create_dataset('dataset4', data=matrix_4)

    # Suppose you want to store a 100GB dataset in a HDF5 file. In this case, 'compression'
    G1.create_dataset('dataset1', data=matrix_1, compression='gzip', compression_opts=7) #1-9 higher values more compression
    G1.create_dataset('dataset2', data=matrix_1, compression='gzip', compression_opts=7)


    # Create friends inside Group2
    G21 = hdf.create_group('/Group2/Friends')
    G21.create_dataset('dataset3', data=matrix_3)
    G22 = hdf.create_group('/Group2/Office')
    G22.create_dataset('dataset3', data=matrix_4)



with h5py.File('./hdf5_groups.h5', 'r') as hdf:
    # Can access dataset1 and dataset4 hierarchically
    items = list(hdf.items())
    print(items)
    G1 = hdf.get('/Group1')
    print(list(G1.items()))
    d1 = G1.get('dataset1')
    d1 = np.array(d1)
    print(d1.shape)
    d4 = G1.get('dataset4')
    d4 = np.array(d4)
    print(d4.shape)

    print(40 * '*')

    # Can access dataset1 and dataset4 directly by providing their paths
    # Method 2 for reading d1 and d4
    d1_prime = hdf.get('Group1/dataset1')
    d1_prime = np.array(d1_prime)
    d4_prime = hdf.get('Group1/dataset4')
    d4_prime = np.array(d4_prime)
    print(d1_prime.shape)
    print(d4_prime.shape)
