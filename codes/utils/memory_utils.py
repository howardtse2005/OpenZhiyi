import gc
import time
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

_nvml_handle = None

def init_nvml():
    global _nvml_handle
    nvmlInit()
    _nvml_handle = nvmlDeviceGetHandleByIndex(0)

def wait_for_memory(min_free, timeout=10):
    """Wait until sufficient GPU memory is available (min_free in bytes)"""
    global _nvml_handle
    if _nvml_handle is None:
        init_nvml()
    start = time.time()
    while True:
        info = nvmlDeviceGetMemoryInfo(_nvml_handle)
        if info.free >= min_free * (1024 ** 2):
            break
        if time.time() - start > timeout:
            print(f"WARNING: Couldn't free {min_free}MB after {timeout}s")
            break
        time.sleep(0.5)

def memory_cleanup(huang_model=None, ben_model=None):
    if huang_model is not None:
        huang_model.zero_grad(set_to_none=True)
    if ben_model is not None:
        ben_model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    wait_for_memory(min_free=10*1024)  # Wait for 10GB free
    print("Memory cleanup complete. GPU memory should be available now.")

def print_memory_stats():
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
