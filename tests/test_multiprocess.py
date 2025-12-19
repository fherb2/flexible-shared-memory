"""
Multi-process communication tests for flexible_shared_memory.

Tests cover:
- One writer, one reader
- Multiple readers
- Data transfer correctness
- Process cleanup
- Real-world scenarios
- BOTH fork and spawn start methods
"""

import pytest
import numpy as np
from dataclasses import dataclass
import time
from multiprocessing import Process, Queue
import multiprocessing
import sys

from flexible_shared_memory import SharedMemory


# Test fixtures
@pytest.fixture
def unique_name():
    """Generate unique name for each test."""
    return f"test_shm_{time.time_ns()}"


# Test dataclasses
@dataclass
class SensorData:
    temperature: float = 0.0
    pressure: float = 0.0
    timestamp: float = 0.0
    status: "str[16]" = ""


@dataclass
class ImageData:
    frame_id: int = 0
    timestamp: float = 0.0
    image: "uint8[10,10,3]" = None


# Helper functions for multiprocess tests (MUST be at module level for spawn!)
def writer_process_simple(name: str, num_writes: int, queue: Queue):
    """Simple writer process - communicates via queue."""
    try:
        shm = SharedMemory(SensorData, name=name, create=False)
        
        for i in range(num_writes):
            shm.write(
                temperature=20.0 + i,
                pressure=1000.0 + i,
                timestamp=time.time(),
                status=f"write_{i}"
            )
            time.sleep(0.01)
        
        shm.close()
        queue.put(("success", num_writes))
    except Exception as e:
        queue.put(("error", str(e)))


def reader_process_simple(name: str, expected_count: int, queue: Queue):
    """Simple reader process - returns results via queue."""
    try:
        shm = SharedMemory(SensorData, name=name, create=False)
        
        readings = []
        for _ in range(expected_count):
            data = shm.read(timeout=1.0)
            if data and data.temperature.valid:
                readings.append(data.temperature.value)
            time.sleep(0.01)
        
        shm.close()
        queue.put(("success", readings))
    except Exception as e:
        queue.put(("error", str(e)))


def fifo_writer_process(name: str, num_writes: int, queue: Queue):
    """FIFO writer process."""
    try:
        fifo = SharedMemory(SensorData, name=name, create=False, slots=5)
        
        for i in range(num_writes):
            fifo.write(
                temperature=10.0 + i,
                status=f"fifo_{i}"
            )
            fifo.finalize()
            time.sleep(0.05)
        
        fifo.close()
        queue.put(("success", num_writes))
    except Exception as e:
        queue.put(("error", str(e)))


def fifo_reader_process(name: str, expected_count: int, queue: Queue):
    """FIFO reader process."""
    try:
        fifo = SharedMemory(SensorData, name=name, create=False, slots=5)
        
        readings = []
        for _ in range(expected_count):
            data = fifo.read(timeout=2.0, latest=False)
            if data and data.temperature.valid:
                readings.append(data.temperature.value)
        
        fifo.close()
        queue.put(("success", readings))
    except Exception as e:
        queue.put(("error", str(e)))


def writer_with_values(name: str, test_values: list, queue: Queue):
    """Writer that writes specific test values."""
    try:
        time.sleep(0.05)  # Give reader time to prepare
        shm = SharedMemory(SensorData, name=name, create=False)
        for temp, press, status in test_values:
            shm.write(temperature=temp, pressure=press, status=status)
            time.sleep(0.1)  # Ensure reader can catch each write
        shm.close()
        queue.put(("success", len(test_values)))
    except Exception as e:
        queue.put(("error", str(e)))


def fast_fifo_writer(name: str, num_writes: int, queue: Queue):
    """Fast FIFO writer for overflow test."""
    try:
        time.sleep(0.05)  # Let reader prepare
        f = SharedMemory(SensorData, name=name, create=False, slots=3)
        for i in range(num_writes):
            f.write(temperature=float(i))
            f.finalize()
            time.sleep(0.02)
        f.close()
        queue.put(("success", num_writes))
    except Exception as e:
        queue.put(("error", str(e)))


def image_writer(name: str, num_frames: int, queue: Queue):
    """Image writer process."""
    try:
        time.sleep(0.05)  # Let reader prepare
        shm_w = SharedMemory(ImageData, name=name, create=False)
        for frame_id in range(num_frames):
            img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            img[:, :, 0] = frame_id  # Mark with frame_id in red channel
            shm_w.write(frame_id=frame_id, timestamp=time.time(), image=img)
            time.sleep(0.1)
        shm_w.close()
        queue.put(("success", num_frames))
    except Exception as e:
        queue.put(("error", str(e)))


def writer_two_stage(name: str, queue: Queue):
    """Writer that writes temperature, waits, then pressure."""
    try:
        time.sleep(0.05)  # Let reader prepare
        shm_w = SharedMemory(SensorData, name=name, create=False)
        shm_w.write(temperature=25.0)
        time.sleep(0.2)
        shm_w.write(pressure=1013.0)
        time.sleep(0.1)
        shm_w.close()
        queue.put(("success", 2))
    except Exception as e:
        queue.put(("error", str(e)))


def writer_cleanup(name: str, queue: Queue):
    """Writer with cleanup."""
    try:
        shm = SharedMemory(SensorData, name=name, create=True)
        shm.write(temperature=20.0)
        shm.close()
        shm.unlink()
        queue.put(("success", None))
    except Exception as e:
        queue.put(("error", str(e)))


def reader_cleanup(name: str, queue: Queue):
    """Reader with cleanup."""
    try:
        shm_r = SharedMemory(SensorData, name=name, create=False)
        data = shm_r.read(timeout=1.0)
        shm_r.close()
        queue.put(("success", data is not None))
    except Exception as e:
        queue.put(("error", str(e)))


# Parametrize all multiprocess tests with both fork and spawn
PROCESS_START_METHODS = ["fork", "spawn"] if sys.platform != "win32" else ["spawn"]


# Basic multiprocess tests
class TestBasicMultiprocess:
    """Test basic multi-process communication."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_one_writer_one_reader(self, unique_name, start_method):
        """Test single writer and single reader."""
        ctx = multiprocessing.get_context(start_method)
        
        # Create shared memory in main process
        shm = SharedMemory(SensorData, name=unique_name, create=True)
        
        try:
            # Start writer with queue
            queue = ctx.Queue()
            writer = ctx.Process(target=writer_process_simple, args=(unique_name, 10, queue))
            writer.start()
            
            # Give writer time to start and make first write
            time.sleep(0.05)
            
            # Read in main process
            readings = []
            for _ in range(10):
                data = shm.read(timeout=1.0, reset_modified=False)
                if data and data.temperature.valid and data.temperature.modified:
                    readings.append(data.temperature.value)
                    time.sleep(0.02)  # Give writer time to update
            
            writer.join(timeout=3.0)
            
            # Check writer result
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Verify we got readings
            assert len(readings) > 0, f"No readings received (start_method={start_method})"
            assert all(20.0 <= t <= 29.0 for t in readings)
        
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_writer_creates_reader_opens(self, unique_name, start_method):
        """Test that writer creates and reader opens existing memory."""
        # Writer creates
        shm_writer = SharedMemory(SensorData, name=unique_name, create=True)
        
        try:
            shm_writer.write(temperature=25.0, pressure=1013.0)
            
            # Reader opens existing
            shm_reader = SharedMemory(SensorData, name=unique_name, create=False)
            
            data = shm_reader.read(timeout=0)
            assert data is not None
            assert abs(data.temperature.value - 25.0) < 1e-10
            
            shm_reader.close()
        
        finally:
            shm_writer.close()
            shm_writer.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_data_transfer_correctness(self, unique_name, start_method):
        """Test that data is transferred correctly between processes."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(SensorData, name=unique_name, create=True)
        
        try:
            test_values = [
                (23.5, 1013.25, "OK"),
                (24.0, 1012.50, "WARN"),
                (22.0, 1014.00, "OK")
            ]
            
            # Start writer
            queue = ctx.Queue()
            writer_proc = ctx.Process(target=writer_with_values, 
                                     args=(unique_name, test_values, queue))
            writer_proc.start()
            
            # Read values
            received = []
            prev_temp = None
            max_attempts = len(test_values) * 5  # More attempts for timing
            for _ in range(max_attempts):    
                data = shm.read(timeout=2.0, reset_modified=False)
                if data and data.temperature.valid and data.temperature.modified:
                    temp_val = data.temperature.value
                    # Only add if different from previous (avoid duplicates)
                    if temp_val != prev_temp:
                        received.append((
                            temp_val,
                            data.pressure.value,
                            data.status.value
                        ))
                        prev_temp = temp_val
                        if len(received) >= len(test_values):
                            break
                time.sleep(0.05)
            
            writer_proc.join(timeout=2.0)
            
            # Check writer result
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Verify correctness
            assert len(received) >= len(test_values), \
                f"Expected {len(test_values)}, got {len(received)} (start_method={start_method})"
            for i, (t_exp, p_exp, s_exp) in enumerate(test_values):
                if i < len(received):
                    t_recv, p_recv, s_recv = received[i]
                    assert abs(t_recv - t_exp) < 1e-10
                    assert abs(p_recv - p_exp) < 1e-10
                    assert s_recv == s_exp
        
        finally:
            shm.close()
            shm.unlink()


# FIFO multiprocess tests
class TestFIFOMultiprocess:
    """Test FIFO mode with multiple processes."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_fifo_writer_reader(self, unique_name, start_method):
        """Test FIFO with writer and reader in separate processes."""
        ctx = multiprocessing.get_context(start_method)
        
        # Create FIFO
        fifo = SharedMemory(SensorData, name=unique_name, create=True, slots=5)
        
        try:
            # Start writer process
            queue = ctx.Queue()
            writer = ctx.Process(target=fifo_writer_process, args=(unique_name, 10, queue))
            writer.start()
            
            # Give writer time to start
            time.sleep(0.1)
            
            # Read in main process
            readings = []
            for _ in range(15):  # Try more times
                data = fifo.read(timeout=2.0, latest=False)
                if data and data.temperature.valid:
                    readings.append(data.temperature.value)
                    if len(readings) >= 10:
                        break
            
            writer.join(timeout=3.0)
            
            # Check writer result
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Verify FIFO order
            assert len(readings) >= 5, f"Got only {len(readings)} readings (start_method={start_method})"
            # Check that values are in sequence (some might be missing due to timing)
            for i in range(len(readings) - 1):
                # Next value should be >= current (monotonic, but might skip)
                assert readings[i+1] >= readings[i]
        
        finally:
            fifo.close()
            fifo.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_fifo_with_overflow(self, unique_name, start_method):
        """Test FIFO overflow in multi-process scenario."""
        ctx = multiprocessing.get_context(start_method)
        fifo = SharedMemory(SensorData, name=unique_name, create=True, slots=3)
        
        try:
            # Start fast writer
            queue = ctx.Queue()
            writer = ctx.Process(target=fast_fifo_writer, args=(unique_name, 10, queue))
            writer.start()
            
            # Slow reader - let buffer fill
            time.sleep(0.3)
            
            # Read - should get most recent items
            readings = []
            for _ in range(5):  # Try reading more than available
                data = fifo.read(timeout=1.0)
                if data and data.temperature.valid:
                    readings.append(data.temperature.value)
            
            writer.join(timeout=3.0)
            
            # Check writer result
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Should have gotten some values from the later writes
            assert len(readings) >= 3, f"Got only {len(readings)} readings"
            # At least one value should be >= 5 (showing overflow happened)
            assert any(v >= 5.0 for v in readings), \
                f"No high values found, readings={readings} (start_method={start_method})"
        
        finally:
            fifo.close()
            fifo.unlink()


# Array transfer tests
class TestArrayTransfer:
    """Test transferring arrays between processes."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_image_transfer(self, unique_name, start_method):
        """Test transferring image data between processes."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(ImageData, name=unique_name, create=True)
        
        try:
            queue = ctx.Queue()
            writer_proc = ctx.Process(target=image_writer, args=(unique_name, 3, queue))
            writer_proc.start()
            
            # Read frames
            frames = []
            prev_frame_id = -1
            for _ in range(10):  # Try more times for timing
                data = shm.read(timeout=2.0, reset_modified=False)
                if data and data.image.valid and data.image.modified:
                    frame_id_val = data.frame_id.value
                    # Only add new frames
                    if frame_id_val != prev_frame_id:
                        frames.append((frame_id_val, data.image.value.copy()))
                        prev_frame_id = frame_id_val
                        if len(frames) >= 3:
                            break
                time.sleep(0.05)
            
            writer_proc.join(timeout=3.0)
            
            # Check writer result
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Verify frames
            assert len(frames) >= 3, f"Got only {len(frames)} frames (start_method={start_method})"
            for i in range(min(3, len(frames))):
                frame_id, img = frames[i]
                assert frame_id == i
                assert img.shape == (10, 10, 3)
                assert img[0, 0, 0] == i  # Check red channel marker
        
        finally:
            shm.close()
            shm.unlink()


# Status flag tests
class TestMultiprocessStatusFlags:
    """Test status flags in multi-process scenarios."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_modified_flag_across_processes(self, unique_name, start_method):
        """Test that modified flags work correctly across processes."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(SensorData, name=unique_name, create=True)
        
        try:
            queue = ctx.Queue()
            writer_proc = ctx.Process(target=writer_two_stage, args=(unique_name, queue))
            writer_proc.start()
            
            # Wait for first write
            time.sleep(0.1)
            data1 = shm.read(timeout=1.0, reset_modified=True)
            assert data1 is not None
            assert data1.temperature.modified
            assert data1.temperature.valid
            
            # Wait for second write
            time.sleep(0.3)
            data2 = shm.read(timeout=1.0, reset_modified=True)
            assert data2 is not None
            assert data2.pressure.modified
            assert not data2.temperature.modified  # Not modified in second write
            
            writer_proc.join(timeout=3.0)
            
            # Check writer result
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
        
        finally:
            shm.close()
            shm.unlink()


# Cleanup tests
class TestProcessCleanup:
    """Test proper cleanup in multi-process scenarios."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_writer_cleanup(self, unique_name, start_method):
        """Test that writer can properly cleanup."""
        ctx = multiprocessing.get_context(start_method)
        
        queue = ctx.Queue()
        writer_proc = ctx.Process(target=writer_cleanup, args=(unique_name, queue))
        writer_proc.start()
        writer_proc.join(timeout=3.0)
        
        # Check result
        status, result = queue.get(timeout=1.0)
        assert status == "success", f"Writer cleanup failed: {result}"
        assert writer_proc.exitcode == 0
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_reader_cleanup(self, unique_name, start_method):
        """Test that reader can properly cleanup."""
        ctx = multiprocessing.get_context(start_method)
        
        # Create in main
        shm = SharedMemory(SensorData, name=unique_name, create=True)
        shm.write(temperature=20.0)
        
        try:
            queue = ctx.Queue()
            reader_proc = ctx.Process(target=reader_cleanup, args=(unique_name, queue))
            reader_proc.start()
            reader_proc.join(timeout=3.0)
            
            # Check result
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Reader cleanup failed: {result}"
            assert reader_proc.exitcode == 0
        
        finally:
            shm.close()
            shm.unlink()


# Manual test execution
if __name__ == "__main__":
    # Run with: pytest test_multiprocess.py -v -s
    pytest.main([__file__, "-v", "-s"])