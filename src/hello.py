from multiprocessing import Process, current_process
import os

def worker(rank: int, world_size: int):
    """Each worker prints its rank — mirrors MPI_Comm_rank / MPI_Comm_size."""
    pid = os.getpid()
    print(f"Hello from rank {rank}/{world_size} (PID {pid})")

if __name__ == "__main__":
    WORLD_SIZE = 4

    processes = []
    for rank in range(WORLD_SIZE):
        p = Process(target=worker, args=(rank, WORLD_SIZE))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"\nAll {WORLD_SIZE} ranks finished.")