from multiprocessing import Process, Queue
import numpy as np

def rank_0(send_queue: Queue):
    """Rank 0 sends an array to rank 1 — mirrors MPI_Send."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Rank 0 sending: {data}")
    send_queue.put(data)

def rank_1(recv_queue: Queue):
    """Rank 1 receives the array — mirrors MPI_Recv."""
    data = recv_queue.get()
    print(f"Rank 1 received: {data}")
    print(f"Rank 1 sum: {data.sum()}")

if __name__ == "__main__":
    queue = Queue()

    p0 = Process(target=rank_0, args=(queue,))
    p1 = Process(target=rank_1, args=(queue,))

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    print("\nPoint-to-point communication complete.")