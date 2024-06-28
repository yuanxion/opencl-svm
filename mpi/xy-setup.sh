
sudo apt install mpich
    $ mpirun --version
    HYDRA build details:
        Version:                                 4.0

source build_mpi.sh test_ocl_mpi.cpp
$ mpirun -np 2 ./app
    Data transfer successful

source build_mpi.sh hello_world-mpi.cpp
$ mpirun -np 4 ./app
    world_size: 4
    world_rank: 1
    world_size: 4
    world_rank: 2
    world_size: 4
    world_rank: 3
    world_size: 4
    world_rank: 0
    Process 0 sent number 427 to process 1
    Process 0 sent number 427 to process 2
    Process 0 sent number 427 to process 3
    Process 1 received number 427 from process 0
    Process 2 received number 427 from process 0
    Process 3 received number 427 from process 0
    MPI_Finalized, Done
    MPI_Finalized, Done
    MPI_Finalized, Done
    MPI_Finalized, Done
