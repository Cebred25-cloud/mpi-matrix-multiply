import numpy as np
from multiprocessing import Process, Queue
from loguru import logger

# ── Distributed matrix multiply (1D row decomposition) ───────────────────────
def worker(rank: int, local_A: np.ndarray, B: np.ndarray,
           result_queue: Queue):
    """
    Each worker:
    1. Receives its chunk of rows from A  (mirrors MPI_Scatter)
    2. Has full matrix B                  (mirrors MPI_Bcast)
    3. Computes local_A @ B
    4. Sends result back to rank 0        (mirrors MPI_Gather)
    """
    local_result = local_A @ B
    logger.info(f"Rank {rank} computed block: {local_result.shape}")
    result_queue.put((rank, local_result))

def distributed_matmul(A: np.ndarray, B: np.ndarray,
                        world_size: int) -> np.ndarray:
    N = A.shape[0]
    assert N % world_size == 0, "N must be divisible by world_size"
    rows_per_rank = N // world_size

    result_queue = Queue()
    processes    = []

    # scatter rows and launch workers
    for rank in range(world_size):
        start   = rank * rows_per_rank
        end     = start + rows_per_rank
        local_A = A[start:end].copy()
        p = Process(target=worker,
                    args=(rank, local_A, B, result_queue))
        processes.append(p)
        p.start()

    # gather results BEFORE joining
    results = {}
    for _ in range(world_size):
        rank, local_result = result_queue.get()
        results[rank] = local_result

    for p in processes:
        p.join()

    # assemble in rank order
    C = np.vstack([results[r] for r in range(world_size)])
    return C

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    WORLD_SIZE = 4
    N          = 512

    logger.info(f"Generating {N}x{N} matrices...")
    np.random.seed(42)
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    logger.info(f"Running distributed matmul ({WORLD_SIZE} workers)...")
    C_distributed = distributed_matmul(A, B, WORLD_SIZE)

    # correctness check
    logger.info("Verifying correctness...")
    C_reference = A @ B
    is_correct  = np.allclose(C_distributed, C_reference, atol=1e-4)

    if is_correct:
        logger.success("Correctness check PASSED — matches numpy reference")
    else:
        logger.error("Correctness check FAILED")
        diff = np.max(np.abs(C_distributed - C_reference))
        logger.error(f"Max difference: {diff}")

    logger.info(f"Result shape: {C_distributed.shape}")
    logger.info(f"Result sample [0:3, 0:3]:\n{C_distributed[:3, :3]}")