from multiprocessing import Process, Queue
import numpy as np

def worker(rank: int, world_size: int, bcast_queue: Queue, reduce_queue: Queue):
    """
    Each rank:
    1. Receives a broadcast value from rank 0  (mirrors MPI_Bcast)
    2. Squares it locally
    3. Sends result back to rank 0             (mirrors MPI_Reduce with MPI.SUM)
    """
    # ── Broadcast: receive value from rank 0 ─────────────────────────────────
    value = bcast_queue.get()
    print(f"Rank {rank} received broadcast value: {value}")

    # ── Local computation ─────────────────────────────────────────────────────
    local_result = value ** 2
    print(f"Rank {rank} computed: {value}^2 = {local_result}")

    # ── Reduce: send result back to rank 0 ────────────────────────────────────
    reduce_queue.put(local_result)

def rank_0_coordinator(world_size: int, bcast_queue: Queue, reduce_queue: Queue):
    """
    Rank 0:
    1. Broadcasts a value to all ranks
    2. Collects and sums all results (reduce with SUM)
    """
    broadcast_value = 3.0
    print(f"Rank 0 broadcasting value: {broadcast_value}\n")

    # broadcast — put one copy per worker
    for _ in range(world_size):
        bcast_queue.put(broadcast_value)

    # reduce — collect all results and sum
    total = 0.0
    for _ in range(world_size):
        total += reduce_queue.get()

    print(f"\nRank 0 reduced sum: {total}")
    print(f"Expected ({world_size} ranks × {broadcast_value}^2): "
          f"{world_size * broadcast_value**2}")

if __name__ == "__main__":
    WORLD_SIZE = 4

    bcast_queue  = Queue()
    reduce_queue = Queue()

    # start workers
    workers = []
    for rank in range(WORLD_SIZE):
        p = Process(target=worker,
                    args=(rank, WORLD_SIZE, bcast_queue, reduce_queue))
        workers.append(p)
        p.start()

    # run coordinator in main process
    rank_0_coordinator(WORLD_SIZE, bcast_queue, reduce_queue)

    for p in workers:
        p.join()

    print("\nBroadcast + Reduce complete.")