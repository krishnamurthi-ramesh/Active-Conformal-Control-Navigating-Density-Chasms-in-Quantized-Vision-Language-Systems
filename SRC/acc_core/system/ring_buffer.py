import numpy as np
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory
from typing import Optional
import struct
import time

# PROTOCOL CONSTANTS
BUFFER_SIZE = 1024 * 1024  # 1 MB (Plenty for state vectors)
FLAG_NORMAL = 0
FLAG_DRIFT_DETECTED = 1
FLAG_INTERVENTION_active = 2

class ACCOpsBridge:
    """
    The Nervous System: Zero-copy link between GPU Student and CPU Oracle.
    Uses /dev/shm for microsecond-latency communication.
    """
    shm: SharedMemory
    buffer: memoryview

    def __init__(self, name: str = "acc_bridge", create: bool = False):
        self.name = name
        self.size = BUFFER_SIZE
        try:
            if create:
                # Oracle creates the memory
                try:
                    self.shm = shared_memory.SharedMemory(create=True, size=self.size, name=self.name)
                except FileExistsError:
                    self.shm = shared_memory.SharedMemory(create=False, name=self.name)
                
                # Initialize Header: [Timestamp (f64) | Flag (i32) | Step (i32)]
                buf = self.shm.buf
                assert buf is not None
                self.buffer = buf
                self.buffer[:16] = bytearray(16) 
            else:
                # Student attaches to existing memory
                self.shm = shared_memory.SharedMemory(create=False, name=self.name)
                buf = self.shm.buf
                assert buf is not None
                self.buffer = buf
        except Exception as e:
            print(f"CRITICAL: Failed to access Shared Memory '{name}'. Did the Oracle start first?")
            raise e

    def close(self):
        self.shm.close()
        if hasattr(self, 'unlink'):
            self.shm.unlink()

    # --- STUDENT METHODS (GPU) ---
    def write_state(self, step, z_t_vector):
        """Student writes the compressed state z_t to the buffer."""
        # Layout: [Timestamp(8)] [Flag(4)] [Step(4)] [VectorData(...)]
        timestamp = time.time()
        header = struct.pack('dii', timestamp, FLAG_NORMAL, step)
        
        # 1. Write Header (Offset 0-16)
        self.buffer[:16] = header
        
        # 2. Write Data (Offset 16+)
        # Ensure vector is fp32 for transport efficiency
        data_bytes = z_t_vector.astype(np.float32).tobytes()
        self.buffer[16:16+len(data_bytes)] = data_bytes

    def check_for_intervention(self):
        """Student checks if Oracle has raised the RED FLAG."""
        # Read just the integer flag at offset 8
        flag = struct.unpack('i', self.buffer[8:12])[0]
        return flag == FLAG_DRIFT_DETECTED

    # --- ORACLE METHODS (CPU) ---
    def read_latest_state(self):
        """Oracle polls for new states."""
        # Read Header
        timestamp, flag, step = struct.unpack('dii', self.buffer[:16])
        return timestamp, flag, step

    def trigger_intervention(self):
        """Oracle slams the brakes."""
        # Overwrite Flag at Offset 8 with DRIFT_DETECTED
        self.buffer[8:12] = struct.pack('i', FLAG_DRIFT_DETECTED)
        
    def clear_intervention(self):
        """Oracle releases the brakes."""
        self.buffer[8:12] = struct.pack('i', FLAG_NORMAL)

# Validation Test Block
if __name__ == "__main__":
    print("Testing Shared Memory Access...")
    try:
        bridge = ACCOpsBridge(create=True)
        print(" Oracle: Memory Segment Created.")
        bridge.close()
        bridge.shm.unlink()
        print(" Oracle: Cleanup Successful.")
    except Exception as e:
        print(f" Failure: {e}")
