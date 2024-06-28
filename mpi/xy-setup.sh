source build_mpi.sh test_ocl_mpi.cpp

sudo apt install mpich
    $ mpirun --version
    HYDRA build details:
        Version:                                 4.0

$ mpirun -np 2 ./app
Data transfer successful

