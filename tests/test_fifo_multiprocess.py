"""
FIFO mode tests with auto-detection.

Tests that readers automatically detect FIFO configuration
and handle slots correctly without explicit parameter.
"""

import pytest
import numpy as np
from dataclasses import dataclass
import multiprocessing
from multiprocessing import Queue
import sys

from flexible_shared_memory import SharedMemory


PROCESS_START_METHODS = ["fork", "spawn"] if sys.platform != "win32" else ["spawn"]


@dataclass
class FIFOData:
    value: float = 0.0
    count: int = 0


@dataclass
class MixedData:
    position: float = 0.0
    name: "str[16]" = ""


@dataclass
class ArrayData:
    data: "float32[5]" = None


# Helper functions - ALL use ATTACH mode (readers auto-detect!)
def write_and_finalize_once(name: str, value: float, count: int, queue: Queue):
    """Write once and finalize."""
    try:
        fifo = SharedMemory(name)  # ATTACH - Auto-detects slots!
        fifo.write(value=value, count=count)
        fifo.finalize()
        fifo.close()
        queue.put(("success", None))
    except Exception as e:
        queue.put(("error", str(e)))


def write_sequence(name: str, start: int, count: int, queue: Queue):
    """Write a sequence of values."""
    try:
        fifo = SharedMemory(name)  # ATTACH - Auto-detects!
        for i in range(start, start + count):
            fifo.write(value=float(i), count=i)
            fifo.finalize()
        fifo.close()
        queue.put(("success", count))
    except Exception as e:
        queue.put(("error", str(e)))


def write_overflow_sequence(name: str, total: int, queue: Queue):
    """Write sequence that overflows buffer."""
    try:
        fifo = SharedMemory(name)  # ATTACH
        for i in range(total):
            fifo.write(value=float(i))
            fifo.finalize()
        fifo.close()
        queue.put(("success", total))
    except Exception as e:
        queue.put(("error", str(e)))


def write_mixed_sequence(name: str, messages: list, queue: Queue):
    """Write sequence with mixed types."""
    try:
        fifo = SharedMemory(name)  # ATTACH
        for i, msg in enumerate(messages):
            fifo.write(position=float(i), name=msg)
            fifo.finalize()
        fifo.close()
        queue.put(("success", len(messages)))
    except Exception as e:
        queue.put(("error", str(e)))


def write_array_sequence(name: str, count: int, queue: Queue):
    """Write sequence with arrays."""
    try:
        fifo = SharedMemory(name)  # ATTACH
        for i in range(count):
            arr = np.ones(5, dtype=np.float32) * i
            fifo.write(data=arr)
            fifo.finalize()
        fifo.close()
        queue.put(("success", count))
    except Exception as e:
        queue.put(("error", str(e)))


class TestFIFOBasics:
    """Test basic FIFO operations."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_create_fifo(self, start_method):
        """Test creating FIFO."""
        fifo = SharedMemory(FIFOData, slots=5)  # CREATE FIFO
        
        try:
            assert fifo.is_fifo
            assert fifo.slots == 5
        finally:
            fifo.close()
            fifo.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_write_and_finalize(self, start_method):
        """Test write and finalize from subprocess."""
        ctx = multiprocessing.get_context(start_method)
        fifo = SharedMemory(FIFOData, slots=5)  # CREATE FIFO
        name = fifo.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_and_finalize_once, 
                             args=(name, 1.0, 1, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = fifo.read(timeout=1.0)
            assert data is not None
            assert abs(data.value.value - 1.0) < 1e-10
            assert data.count.value == 1
        finally:
            fifo.close()
            fifo.unlink()

    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_read_empty_fifo_timeout(self, start_method):
        """Test reading from empty FIFO returns None after timeout."""
        fifo = SharedMemory(FIFOData, slots=5)  # CREATE FIFO
        
        try:
            # Read from empty FIFO with timeout
            data = fifo.read(timeout=0.1)
            assert data is None, "Reading from empty FIFO should return None after timeout"
            
            # Write one item
            fifo.write(value=1.0, count=1)
            fifo.finalize()
            
            # Read it
            data = fifo.read(timeout=1.0)
            assert data is not None
            assert abs(data.value.value - 1.0) < 1e-10
            
            # Now empty again
            data = fifo.read(timeout=0.1)
            assert data is None, "Should return None when FIFO becomes empty"
        finally:
            fifo.close()
            fifo.unlink()


class TestFIFOOrdering:
    """Test FIFO read ordering."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_read_in_order(self, start_method):
        """Test reading data in FIFO order."""
        ctx = multiprocessing.get_context(start_method)
        fifo = SharedMemory(FIFOData, slots=5)  # CREATE FIFO
        name = fifo.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_sequence, 
                             args=(name, 0, 3, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Read in order
            for i in range(3):
                data = fifo.read(timeout=1.0, latest=False)
                assert data is not None, f"Failed to read item {i}"
                assert abs(data.value.value - float(i)) < 1e-10
                assert data.count.value == i
        finally:
            fifo.close()
            fifo.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_read_latest_skips_old(self, start_method):
        """Test latest=True skips to newest."""
        ctx = multiprocessing.get_context(start_method)
        fifo = SharedMemory(FIFOData, slots=5)  # CREATE FIFO
        name = fifo.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_sequence, 
                             args=(name, 0, 5, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Read with latest=True
            data = fifo.read(timeout=1.0, latest=True)
            assert data is not None
            assert abs(data.value.value - 4.0) < 1e-10
        finally:
            fifo.close()
            fifo.unlink()


class TestFIFOOverflow:
    """Test FIFO overflow behavior."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_overflow_overwrites_oldest(self, start_method):
        """Test FIFO overwrites oldest when full."""
        ctx = multiprocessing.get_context(start_method)
        fifo = SharedMemory(FIFOData, slots=3)  # CREATE FIFO
        name = fifo.name
        
        try:
            # Write 5 items into 3-slot FIFO
            queue = ctx.Queue()
            proc = ctx.Process(target=write_overflow_sequence, 
                             args=(name, 5, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Should get 2, 3, 4
            data = fifo.read(timeout=1.0)
            assert data is not None
            assert abs(data.value.value - 2.0) < 1e-10
            
            data = fifo.read(timeout=1.0)
            assert abs(data.value.value - 3.0) < 1e-10
            
            data = fifo.read(timeout=1.0)
            assert abs(data.value.value - 4.0) < 1e-10
            
            # Empty
            data = fifo.read(timeout=0)
            assert data is None
        finally:
            fifo.close()
            fifo.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_continuous_overflow(self, start_method):
        """Test continuous overflow."""
        ctx = multiprocessing.get_context(start_method)
        fifo = SharedMemory(FIFOData, slots=3)  # CREATE FIFO
        name = fifo.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_overflow_sequence, 
                             args=(name, 10, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Should have last 3: 7, 8, 9
            for expected in [7, 8, 9]:
                data = fifo.read(timeout=1.0)
                assert data is not None, f"Failed to read {expected}"
                assert abs(data.value.value - float(expected)) < 1e-10
        finally:
            fifo.close()
            fifo.unlink()

    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_overflow_flag_set(self, start_method):
        """Test that overflow flag is set when FIFO overflows."""
        ctx = multiprocessing.get_context(start_method)
        fifo = SharedMemory(FIFOData, slots=3)  # CREATE FIFO
        name = fifo.name
        
        try:
            # Write 5 items into 3-slot FIFO (causes overflow)
            queue = ctx.Queue()
            proc = ctx.Process(target=write_overflow_sequence, 
                             args=(name, 5, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Read remaining values and check overflow flag
            overflow_detected = False
            for _ in range(3):
                data = fifo.read(timeout=1.0)
                assert data is not None, "Failed to read data after overflow"
                
                # Check if overflow flag is set on any field
                if data.value.overflow or data.count.overflow:
                    overflow_detected = True
            
            assert overflow_detected, "overflow flag should be set after FIFO overflow (start_method={start_method})"
        finally:
            fifo.close()
            fifo.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_overflow_flag_details(self, start_method):
        """Test overflow flag on specific fields."""
        fifo = SharedMemory(FIFOData, slots=2)  # CREATE FIFO
        
        try:
            # Write 4 items to 2-slot FIFO
            for i in range(4):
                fifo.write(value=float(i), count=i)
                fifo.finalize()
            
            # Read remaining items (should be 2, 3)
            data1 = fifo.read(timeout=1.0)
            assert data1 is not None
            assert abs(data1.value.value - 2.0) < 1e-10
            assert data1.value.overflow, "First remaining item should have overflow flag"
            
            data2 = fifo.read(timeout=1.0)
            assert data2 is not None
            assert abs(data2.value.value - 3.0) < 1e-10
            assert data2.value.overflow, "Second remaining item should have overflow flag"
            
            # FIFO should be empty now
            data3 = fifo.read(timeout=0.1)
            assert data3 is None, "FIFO should be empty"
        finally:
            fifo.close()
            fifo.unlink()

class TestFIFOSlotCounts:
    """Test different slot counts."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    @pytest.mark.parametrize("num_slots", [2, 5, 10])
    def test_fifo_various_slots(self, start_method, num_slots):
        """Test FIFO with various slot counts."""
        ctx = multiprocessing.get_context(start_method)
        fifo = SharedMemory(FIFOData, slots=num_slots)  # CREATE FIFO
        name = fifo.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_sequence, 
                             args=(name, 0, num_slots, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Read all
            for i in range(num_slots):
                data = fifo.read(timeout=1.0)
                assert data is not None, f"Failed to read item {i}"
                assert abs(data.value.value - float(i)) < 1e-10
        finally:
            fifo.close()
            fifo.unlink()


class TestFIFOMixedTypes:
    """Test FIFO with mixed types."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_fifo_with_strings(self, start_method):
        """Test FIFO with strings."""
        ctx = multiprocessing.get_context(start_method)
        fifo = SharedMemory(MixedData, slots=3)  # CREATE FIFO
        name = fifo.name
        
        try:
            messages = ["first", "second", "third"]
            queue = ctx.Queue()
            proc = ctx.Process(target=write_mixed_sequence, 
                             args=(name, messages, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            for i, msg in enumerate(messages):
                data = fifo.read(timeout=1.0)
                assert data is not None, f"Failed to read {i}"
                assert abs(data.position.value - float(i)) < 1e-10
                assert data.name.value == msg
        finally:
            fifo.close()
            fifo.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_fifo_with_arrays(self, start_method):
        """Test FIFO with arrays."""
        ctx = multiprocessing.get_context(start_method)
        fifo = SharedMemory(ArrayData, slots=3)  # CREATE FIFO
        name = fifo.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_array_sequence, 
                             args=(name, 3, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            for i in range(3):
                data = fifo.read(timeout=1.0)
                assert data is not None, f"Failed to read {i}"
                expected = np.ones(5, dtype=np.float32) * i
                np.testing.assert_array_almost_equal(data.data.value, expected)
        finally:
            fifo.close()
            fifo.unlink()


class TestFIFORestrictions:
    """Test FIFO mode restrictions."""
    
    def test_reset_modified_not_allowed_in_fifo(self):
        """Test that reset_modified raises error in FIFO mode."""
        fifo = SharedMemory(FIFOData, slots=3)  # CREATE FIFO
        
        try:
            # Write some data
            fifo.write(value=1.0, count=1)
            fifo.finalize()
            
            # Try to read with reset_modified - should fail
            with pytest.raises(ValueError, match="reset_modified only supported in single-slot mode"):
                fifo.read(timeout=1.0, reset_modified=True)
            
            # Normal read should work
            data = fifo.read(timeout=1.0, reset_modified=False)
            assert data is not None
            assert abs(data.value.value - 1.0) < 1e-10
        finally:
            fifo.close()
            fifo.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_finalize_required_in_fifo(self, start_method):
        """Test that write without finalize doesn't publish in FIFO mode."""
        ctx = multiprocessing.get_context(start_method)
        fifo = SharedMemory(FIFOData, slots=3)  # CREATE FIFO
        
        try:
            # Write WITHOUT finalize
            fifo.write(value=1.0, count=1)
            # Don't call finalize()
            
            # Try to read - should timeout (nothing published)
            data = fifo.read(timeout=0.2)
            assert data is None, "write() without finalize() should not publish data in FIFO mode"
            
            # Now finalize
            fifo.finalize()
            
            # Should be readable now
            data = fifo.read(timeout=1.0)
            assert data is not None
            assert abs(data.value.value - 1.0) < 1e-10
        finally:
            fifo.close()
            fifo.unlink()

    def test_finalize_when_not_dirty(self):
        """Test that finalize() returns early when buffer not dirty."""
        fifo = SharedMemory(FIFOData, slots=3)  # CREATE FIFO
        
        try:
            # Call finalize without writing - should return early
            fifo.finalize()  # No-op, buffer not dirty
            
            # Read should return None (no data)
            data = fifo.read(timeout=0.1)
            assert data is None, "finalize() without write should not publish data"
            
            # Now write and finalize properly
            fifo.write(value=1.0)
            fifo.finalize()
            
            data = fifo.read(timeout=1.0)
            assert data is not None
            assert abs(data.value.value - 1.0) < 1e-10
        finally:
            fifo.close()
            fifo.unlink()
    
    def test_fifo_metadata_on_single_slot(self):
        """Test that FIFO metadata functions handle single-slot mode."""
        # This tests the guards in _get_fifo_metadata and _set_fifo_metadata
        single = SharedMemory(FIFOData, slots=1)  # CREATE single-slot
        
        try:
            # These should return (0, 0, 0) for single-slot
            write_idx, read_idx, count = single._get_fifo_metadata()
            assert write_idx == 0
            assert read_idx == 0
            assert count == 0
            
            # This should return without doing anything
            single._set_fifo_metadata(1, 1, 1)  # Should be no-op
            
            # Verify it didn't actually change anything
            write_idx, read_idx, count = single._get_fifo_metadata()
            assert write_idx == 0
            assert read_idx == 0
            assert count == 0
        finally:
            single.close()
            single.unlink()
            

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
