#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::cout << "world_size: " << world_size << std::endl;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "world_rank: " << world_rank << std::endl;

    if (world_rank == 0) {
        int number = 427;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "Process " << world_rank << " sent number " << number << " to process 1" << std::endl;
        MPI_Send(&number, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
        std::cout << "Process " << world_rank << " sent number " << number << " to process 2" << std::endl;
        MPI_Send(&number, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);
        std::cout << "Process " << world_rank << " sent number " << number << " to process 3" << std::endl;
    // } else if (world_rank == 1) {
    } else {
        int number;
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process " << world_rank << " received number " << number << " from process 0" << std::endl;
    }

    MPI_Finalize();
    std::cout << "MPI_Finalized, Done" << std::endl;
    return 0;
}
