"""
Multi-process communication tests for flexible_shared_memory.

Tests the new self-describing header functionality:
- Readers auto-detect slots from header
- Hash validation ensures data integrity
- Works with both fork and spawn
"""

import pytest
import numpy as np
from dataclasses import dataclass
import time
from multiprocessing import Process, Queue
import multiprocessing
import sys

from flexible_shared_memory import SharedMemory


@pytest.fixture
def unique_name():
    """Generate unique name for each test."""
    return f"test_shm_{time.time_ns()}"


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


# Helper functions (NO slots parameter needed!)
def writer_process_simple(name: str, num_writes: int, queue: Queue):
    """Simple writer - reader will auto-detect configuration."""
    try:
        # Reader auto-detects, so just open
        shm = SharedMemory(SensorData, name=name)
        
        queue.put(("ready", None))
        
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


def fifo_writer_process(name: str, num_writes: int, queue: Queue):
    """FIFO writer - reader auto-detects slots."""
    try:
        fifo = SharedMemory(SensorData, name=name)  # Auto-detects slots=5!
        
        for i in range(num_writes):
            fifo.write(temperature=10.0 + i, status=f"fifo_{i}")
            fifo.finalize()
            time.sleep(0.05)
        
        fifo.close()
        queue.put(("success", num_writes))
    except Exception as e:
        queue.put(("error", str(e)))


def writer_with_values(name: str, test_values: list, queue: Queue):
    """Writer with specific test values."""
    try:
        time.sleep(0.05)
        shm = SharedMemory(SensorData, name=name)
        for temp, press, status in test_values:
            shm.write(temperature=temp, pressure=press, status=status)
            time.sleep(0.1)
        shm.close()
        queue.put(("success", len(test_values)))
    except Exception as e:
        queue.put(("error", str(e)))


def fast_fifo_writer(name: str, num_writes: int, queue: Queue):
    """Fast FIFO writer."""
    try:
        time.sleep(0.05)
        f = SharedMemory(SensorData, name=name)  # Auto-detects slots=3!
        for i in range(num_writes):
            f.write(temperature=float(i))
            f.finalize()
            time.sleep(0.02)
        f.close()
        queue.put(("success", num_writes))
    except Exception as e:
        queue.put(("error", str(e)))


def image_writer(name: str, num_frames: int, queue: Queue):
    """Image writer."""
    try:
        time.sleep(0.05)
        shm_w = SharedMemory(ImageData, name=name)
        for frame_id in range(num_frames):
            img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            img[:, :, 0] = frame_id
            shm_w.write(frame_id=frame_id, timestamp=time.time(), image=img)
            time.sleep(0.1)
        shm_w.close()
        queue.put(("success", num_frames))
    except Exception as e:
        queue.put(("error", str(e)))


def writer_two_stage(name: str, queue: Queue):
    """Two-stage writer."""
    try:
        shm_w = SharedMemory(SensorData, name=name)
        
        queue.put(("ready", None))
        
        time.sleep(0.05)
        shm_w.write(temperature=25.0)
        queue.put(("first_write", None))
        
        time.sleep(0.2)
        shm_w.write(pressure=1013.0)
        queue.put(("second_write", None))
        
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
        shm_r = SharedMemory(SensorData, name=name)  # Auto-detect
        data = shm_r.read(timeout=1.0)
        shm_r.close()
        queue.put(("success", data is not None))
    except Exception as e:
        queue.put(("error", str(e)))


PROCESS_START_METHODS = ["fork", "spawn"] if sys.platform != "win32" else ["spawn"]


class TestBasicMultiprocess:
    """Test basic multi-process communication."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_one_writer_one_reader(self, unique_name, start_method):
        """Test single writer and single reader."""
        ctx = multiprocessing.get_context(start_method)
        
        shm = SharedMemory(SensorData, name=unique_name, create=True)
        
        try:
            queue = ctx.Queue()
            writer = ctx.Process(target=writer_process_simple, args=(unique_name, 10, queue))
            writer.start()
            
            status, _ = queue.get(timeout=5.0)
            assert status == "ready", "Writer failed to start"
            
            time.sleep(0.05)
            
            readings = []
            for _ in range(10):
                data = shm.read(timeout=1.0, reset_modified=False)
                if data and data.temperature.valid and data.temperature.modified:
                    readings.append(data.temperature.value)
                    time.sleep(0.02)
            
            writer.join(timeout=3.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            assert len(readings) > 0, f"No readings received (start_method={start_method})"
            assert all(20.0 <= t <= 29.0 for t in readings)
        
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_writer_creates_reader_opens(self, unique_name, start_method):
        """Test that writer creates and reader opens existing memory."""
        shm_writer = SharedMemory(SensorData, name=unique_name, create=True)
        
        try:
            shm_writer.write(temperature=25.0, pressure=1013.0)
            
            # Reader auto-detects configuration
            shm_reader = SharedMemory(SensorData, name=unique_name)
            
            data = shm_reader.read(timeout=0)
            assert data is not None
            assert abs(data.temperature.value - 25.0) < 1e-10
            
            shm_reader.close()
        
        finally:
            shm_writer.close()
            shm_writer.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_data_transfer_correctness(self, unique_name, start_method):
        """Test correct data transfer between processes."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(SensorData, name=unique_name, create=True)
        
        try:
            test_values = [
                (23.5, 1013.25, "OK"),
                (24.0, 1012.50, "WARN"),
                (22.0, 1014.00, "OK")
            ]
            
            queue = ctx.Queue()
            writer_proc = ctx.Process(target=writer_with_values, 
                                     args=(unique_name, test_values, queue))
            writer_proc.start()
            
            received = []
            prev_temp = None
            max_attempts = len(test_values) * 5
            for _ in range(max_attempts):    
                data = shm.read(timeout=2.0, reset_modified=False)
                if data and data.temperature.valid and data.temperature.modified:
                    temp_val = data.temperature.value
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
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
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

    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_dataclass_mismatch_raises_error(self, unique_name, start_method):
        """Reader with different dataclass should fail hash validation."""
        
        @dataclass
        class WrongData:
            value: float = 0.0
            other: int = 0
        
        shm_writer = SharedMemory(SensorData, name=unique_name, create=True)
        shm_writer.write(temperature=25.0)
        
        try:
            # Should raise ValueError due to hash mismatch
            with pytest.raises(ValueError, match="Dataclass structure mismatch"):
                shm_reader = SharedMemory(WrongData, name=unique_name)
        finally:
            shm_writer.close()
            shm_writer.unlink()
    
    def test_reader_auto_detects_slots(self, unique_name):
        """Reader should auto-detect number of slots from header."""
        
        # Writer creates with 7 slots
        shm_writer = SharedMemory(SensorData, name=unique_name, create=True, slots=7)
        
        # Reader opens WITHOUT specifying slots
        shm_reader = SharedMemory(SensorData, name=unique_name)
        
        # Verify reader detected correct configuration
        assert shm_reader.slots == 7, "Reader should auto-detect slots=7"
        assert shm_reader.is_fifo == True, "Reader should detect FIFO mode"
        
        shm_reader.close()
        shm_writer.close()
        shm_writer.unlink()
    
    def test_reader_auto_detects_single_slot(self, unique_name):
        """Reader should auto-detect single-slot mode."""
        
        # Writer creates single-slot (default)
        shm_writer = SharedMemory(SensorData, name=unique_name, create=True, slots=1)
        
        # Reader opens WITHOUT specifying slots
        shm_reader = SharedMemory(SensorData, name=unique_name)
        
        # Verify reader detected single-slot mode
        assert shm_reader.slots == 1, "Reader should auto-detect slots=1"
        assert shm_reader.is_fifo == False, "Reader should detect single-slot mode"
        
        shm_reader.close()
        shm_writer.close()
        shm_writer.unlink()


class TestFIFOMultiprocess:
    """Test FIFO mode with multiple processes."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_fifo_writer_reader(self, unique_name, start_method):
        """Test FIFO - reader auto-detects slots!"""
        ctx = multiprocessing.get_context(start_method)
        
        fifo = SharedMemory(SensorData, name=unique_name, create=True, slots=5)
        
        try:
            queue = ctx.Queue()
            writer = ctx.Process(target=fifo_writer_process, args=(unique_name, 10, queue))
            writer.start()
            
            time.sleep(0.1)
            
            readings = []
            for _ in range(15):
                data = fifo.read(timeout=2.0, latest=False)
                if data and data.temperature.valid:
                    readings.append(data.temperature.value)
                    if len(readings) >= 10:
                        break
            
            writer.join(timeout=3.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            assert len(readings) >= 5, f"Got only {len(readings)} readings (start_method={start_method})"
            for i in range(len(readings) - 1):
                assert readings[i+1] >= readings[i]
        
        finally:
            fifo.close()
            fifo.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_fifo_with_overflow(self, unique_name, start_method):
        """Test FIFO overflow discards oldest data and sets overflow flag."""
        fifo = SharedMemory(SensorData, name=unique_name, create=True, slots=3)
        
        try:
            # Write 10 values to 3-slot FIFO (will overflow)
            for i in range(10):
                fifo.write(temperature=float(i))
                fifo.finalize()
            
            # Read all remaining values
            readings = []
            overflow_detected = False
            for _ in range(5):
                data = fifo.read(timeout=0.1)
                if data and data.temperature.valid:
                    readings.append(data.temperature.value)
                    if data.temperature.overflow:
                        overflow_detected = True
                else:
                    break
            
            # Should get last 3 values (7, 8, 9) due to overflow
            assert len(readings) == 3, f"Expected 3 values, got {len(readings)} (start_method={start_method})"
            assert readings == [7.0, 8.0, 9.0], f"Expected [7.0, 8.0, 9.0], got {readings}"
            assert overflow_detected, "overflow flag should be set when FIFO overflows"
        
        finally:
            fifo.close()
            fifo.unlink()


class TestArrayTransfer:
    """Test array transfer."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_image_transfer(self, unique_name, start_method):
        """Test image transfer."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(ImageData, name=unique_name, create=True)
        
        try:
            queue = ctx.Queue()
            writer_proc = ctx.Process(target=image_writer, args=(unique_name, 3, queue))
            writer_proc.start()
            
            frames = []
            prev_frame_id = -1
            for _ in range(10):
                data = shm.read(timeout=2.0, reset_modified=False)
                if data and data.image.valid and data.image.modified:
                    frame_id_val = data.frame_id.value
                    if frame_id_val != prev_frame_id:
                        frames.append((frame_id_val, data.image.value.copy()))
                        prev_frame_id = frame_id_val
                        if len(frames) >= 3:
                            break
                time.sleep(0.05)
            
            writer_proc.join(timeout=3.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            assert len(frames) >= 3, f"Got only {len(frames)} frames (start_method={start_method})"
            for frame_id, img in frames[:3]:  # Check first 3 frames received
                assert img.shape == (10, 10, 3)
                # Image content should match frame_id (writer sets img[:,:,0] = frame_id)
                assert img[0, 0, 0] == frame_id, \
                    f"Image content mismatch: expected {frame_id}, got {img[0, 0, 0]}"
        
        finally:
            shm.close()
            shm.unlink()


class TestMultiprocessStatusFlags:
    """Test status flags across processes."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_modified_flag_across_processes(self, unique_name, start_method):
        """Test modified flags work across processes."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(SensorData, name=unique_name, create=True)
        
        try:
            queue = ctx.Queue()
            writer_proc = ctx.Process(target=writer_two_stage, args=(unique_name, queue))
            writer_proc.start()
            
            status, _ = queue.get(timeout=5.0)
            assert status == "ready", "Writer failed to start"
            
            status, _ = queue.get(timeout=2.0)
            assert status == "first_write", "First write failed"
            
            data1 = shm.read(timeout=1.0, reset_modified=True)
            assert data1 is not None, f"No data after first write (start_method={start_method})"
            assert data1.temperature.modified
            assert data1.temperature.valid
            
            status, _ = queue.get(timeout=2.0)
            assert status == "second_write", "Second write failed"
            
            data2 = shm.read(timeout=1.0, reset_modified=True)
            assert data2 is not None, f"No data after second write (start_method={start_method})"
            assert data2.pressure.modified
            assert not data2.temperature.modified
            
            writer_proc.join(timeout=3.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
        
        finally:
            shm.close()
            shm.unlink()


    def test_string_utf8_truncation(self, unique_name):
        """Long UTF-8 strings should be truncated with valid boundaries."""
        
        @dataclass
        class ShortStringData:
            message: "str[10]" = ""
        
        shm = SharedMemory(ShortStringData, name=unique_name, create=True)
        
        try:
            # Write string with emojis (4 bytes each)
            long_emoji = "🎉" * 20  # 20 chars
            shm.write(message=long_emoji)
            
            data = shm.read(timeout=0)
            
            # Should be truncated to 10 chars
            assert data is not None, "Should read data"
            assert len(data.message.value) == 10, f"Should truncate to 10 chars, got {len(data.message.value)}"
            assert data.message.truncated == True, "Should be marked as truncated"
            
            # Should still be valid UTF-8
            data.message.value.encode('utf-8')  # Should not raise
        
        finally:
            shm.close()
            shm.unlink()
    
    def test_concurrent_writers_not_supported(self, unique_name):
        """Document that concurrent writers are not supported (single-writer design)."""
        
        shm = SharedMemory(SensorData, name=unique_name, create=True)
        
        try:
            # This test documents that multi-writer scenarios are undefined behavior
            # In practice, concurrent writes would cause sequence number mismatches
            # and data corruption, but we don't test for that - we just document it
            
            # Write a clean value
            shm.write(temperature=25.0, pressure=1013.0, status="OK")
            data = shm.read(timeout=0)
            
            # Verify single writer works fine
            assert data is not None
            assert data.temperature.value == 25.0
            assert data.pressure.value == 1013.0
            
            # NOTE: Multiple concurrent writers would violate the design assumption
            # and cause undefined behavior. This is documented in the docstring.
        
        finally:
            shm.close()
            shm.unlink()


class TestProcessCleanup:
    """Test process cleanup."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_writer_cleanup(self, unique_name, start_method):
        """Test writer cleanup."""
        ctx = multiprocessing.get_context(start_method)
        
        queue = ctx.Queue()
        writer_proc = ctx.Process(target=writer_cleanup, args=(unique_name, queue))
        writer_proc.start()
        writer_proc.join(timeout=3.0)
        
        status, result = queue.get(timeout=1.0)
        assert status == "success", f"Writer cleanup failed: {result}"
        assert writer_proc.exitcode == 0
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_reader_cleanup(self, unique_name, start_method):
        """Test reader cleanup."""
        ctx = multiprocessing.get_context(start_method)
        
        shm = SharedMemory(SensorData, name=unique_name, create=True)
        shm.write(temperature=20.0)
        
        try:
            queue = ctx.Queue()
            reader_proc = ctx.Process(target=reader_cleanup, args=(unique_name, queue))
            reader_proc.start()
            reader_proc.join(timeout=3.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Reader cleanup failed: {result}"
            assert reader_proc.exitcode == 0
        
        finally:
            shm.close()
            shm.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
