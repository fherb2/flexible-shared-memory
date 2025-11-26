"""
FIFO mode tests for flexible_shared_memory.

Tests cover:
- Write and finalize operations
- Reading in FIFO order
- Reading with latest=True
- FIFO overflow behavior
- Slot management
"""

import pytest
import numpy as np
from dataclasses import dataclass
import time

from flexible_shared_memory import SharedMemory


# Test fixtures
@pytest.fixture
def unique_name():
    """Generate unique name for each test."""
    return f"test_shm_{time.time_ns()}"


@pytest.fixture
def cleanup_list():
    """Track SharedMemory instances for cleanup."""
    instances = []
    yield instances
    for shm in instances:
        try:
            shm.close()
            shm.unlink()
        except:
            pass


# Test dataclasses
@dataclass
class SimpleData:
    value: float = 0.0
    count: int = 0


@dataclass
class MixedData:
    position: float = 0.0
    name: "str[16]" = ""


# Basic FIFO tests
class TestFIFOBasics:
    """Test basic FIFO operations."""
    
    def test_create_fifo(self, unique_name, cleanup_list):
        """Test creating FIFO with multiple slots."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=5, create=True)
        cleanup_list.append(fifo)
        
        assert fifo.is_fifo
        assert fifo.slots == 5
    
    def test_write_and_finalize(self, unique_name, cleanup_list):
        """Test basic write and finalize."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=5, create=True)
        cleanup_list.append(fifo)
        
        fifo.write(value=1.0, count=1)
        fifo.finalize()
        
        data = fifo.read(timeout=0)
        assert data is not None
        assert abs(data.value.value - 1.0) < 1e-10
        assert data.count.value == 1
    
    def test_finalize_required(self, unique_name, cleanup_list):
        """Test that data is not visible without finalize."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=5, create=True)
        cleanup_list.append(fifo)
        
        fifo.write(value=1.0)
        # Don't finalize
        
        data = fifo.read(timeout=0)
        assert data is None  # No data without finalize
    
    def test_multiple_writes_before_finalize(self, unique_name, cleanup_list):
        """Test staging multiple writes before finalize."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=5, create=True)
        cleanup_list.append(fifo)
        
        # Stage multiple writes
        fifo.write(value=1.0)
        fifo.write(count=10)
        fifo.finalize()
        
        data = fifo.read(timeout=0)
        assert data is not None
        assert abs(data.value.value - 1.0) < 1e-10
        assert data.count.value == 10


# FIFO ordering tests
class TestFIFOOrdering:
    """Test FIFO read ordering."""
    
    def test_read_in_order(self, unique_name, cleanup_list):
        """Test reading data in FIFO order."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=5, create=True)
        cleanup_list.append(fifo)
        
        # Write 3 items
        for i in range(3):
            fifo.write(value=float(i), count=i)
            fifo.finalize()
        
        # Read in order
        for i in range(3):
            data = fifo.read(timeout=0, latest=False)
            assert data is not None
            assert abs(data.value.value - float(i)) < 1e-10
            assert data.count.value == i
    
    def test_read_latest_skips_old(self, unique_name, cleanup_list):
        """Test that latest=True skips to newest data."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=5, create=True)
        cleanup_list.append(fifo)
        
        # Write multiple items
        for i in range(5):
            fifo.write(value=float(i))
            fifo.finalize()
        
        # Read with latest=True - should get most recent
        data = fifo.read(timeout=0, latest=True)
        assert data is not None
        assert abs(data.value.value - 4.0) < 1e-10
    
    def test_read_empty_fifo(self, unique_name, cleanup_list):
        """Test reading from empty FIFO."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=5, create=True)
        cleanup_list.append(fifo)
        
        data = fifo.read(timeout=0)
        assert data is None


# FIFO overflow tests
class TestFIFOOverflow:
    """Test FIFO overflow behavior."""
    
    def test_overflow_overwrites_oldest(self, unique_name, cleanup_list):
        """Test that FIFO overwrites oldest when full."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=3, create=True)
        cleanup_list.append(fifo)
        
        # Fill FIFO (3 slots)
        for i in range(3):
            fifo.write(value=float(i))
            fifo.finalize()
        
        # Write 2 more (should overwrite 0 and 1)
        for i in range(3, 5):
            fifo.write(value=float(i))
            fifo.finalize()
        
        # Read - should get 2, 3, 4 (0 and 1 were overwritten)
        data = fifo.read(timeout=0)
        assert abs(data.value.value - 2.0) < 1e-10
        
        data = fifo.read(timeout=0)
        assert abs(data.value.value - 3.0) < 1e-10
        
        data = fifo.read(timeout=0)
        assert abs(data.value.value - 4.0) < 1e-10
        
        # FIFO should be empty now
        data = fifo.read(timeout=0)
        assert data is None
    
    def test_continuous_overflow(self, unique_name, cleanup_list):
        """Test writing many items with overflow."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=3, create=True)
        cleanup_list.append(fifo)
        
        # Write 10 items (will overflow multiple times)
        for i in range(10):
            fifo.write(value=float(i))
            fifo.finalize()
        
        # Should have last 3 items: 7, 8, 9
        for expected in [7, 8, 9]:
            data = fifo.read(timeout=0)
            assert data is not None
            assert abs(data.value.value - float(expected)) < 1e-10


# FIFO with different slot counts
class TestFIFOSlotCounts:
    """Test FIFO with various slot counts."""
    
    def test_fifo_with_2_slots(self, unique_name, cleanup_list):
        """Test FIFO with minimum practical slots."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=2, create=True)
        cleanup_list.append(fifo)
        
        fifo.write(value=1.0)
        fifo.finalize()
        fifo.write(value=2.0)
        fifo.finalize()
        
        data1 = fifo.read(timeout=0)
        data2 = fifo.read(timeout=0)
        
        assert abs(data1.value.value - 1.0) < 1e-10
        assert abs(data2.value.value - 2.0) < 1e-10
    
    def test_fifo_with_10_slots(self, unique_name, cleanup_list):
        """Test FIFO with many slots."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=10, create=True)
        cleanup_list.append(fifo)
        
        # Fill all slots
        for i in range(10):
            fifo.write(value=float(i))
            fifo.finalize()
        
        # Read all
        for i in range(10):
            data = fifo.read(timeout=0)
            assert data is not None
            assert abs(data.value.value - float(i)) < 1e-10


# FIFO modified flag tests
class TestFIFOModifiedFlags:
    """Test modified flag behavior in FIFO."""
    
    def test_modified_in_new_slot(self, unique_name, cleanup_list):
        """Test that modified is set in each new slot."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=5, create=True)
        cleanup_list.append(fifo)
        
        fifo.write(value=1.0, count=1)
        fifo.finalize()
        
        data = fifo.read(timeout=0)
        assert data.value.modified
        assert data.count.modified
    
    def test_partial_write_in_fifo(self, unique_name, cleanup_list):
        """Test partial field updates in FIFO."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=5, create=True)
        cleanup_list.append(fifo)
        
        # First slot - write both
        fifo.write(value=1.0, count=1)
        fifo.finalize()
        
        # Second slot - write only value
        fifo.write(value=2.0)
        fifo.finalize()
        
        # Read first
        data1 = fifo.read(timeout=0)
        assert data1.value.modified
        assert data1.count.modified
        
        # Read second
        data2 = fifo.read(timeout=0)
        assert data2.value.modified
        assert not data2.count.modified  # Not written in this slot


# Error handling tests
class TestFIFOErrors:
    """Test FIFO error conditions."""
    
    def test_finalize_on_single_slot_raises(self, unique_name, cleanup_list):
        """Test that finalize() raises error on single-slot."""
        shm = SharedMemory(SimpleData, name=unique_name, slots=1, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=1.0)
        
        with pytest.raises(RuntimeError):
            shm.finalize()
    
    def test_reset_modified_in_fifo_raises(self, unique_name, cleanup_list):
        """Test that reset_modified raises error in FIFO mode."""
        fifo = SharedMemory(SimpleData, name=unique_name, slots=5, create=True)
        cleanup_list.append(fifo)
        
        fifo.write(value=1.0)
        fifo.finalize()
        
        with pytest.raises(ValueError):
            fifo.read(timeout=0, reset_modified=True)


# Mixed type FIFO tests
class TestFIFOMixedTypes:
    """Test FIFO with mixed data types."""
    
    def test_fifo_with_strings(self, unique_name, cleanup_list):
        """Test FIFO with string fields."""
        fifo = SharedMemory(MixedData, name=unique_name, slots=3, create=True)
        cleanup_list.append(fifo)
        
        messages = ["first", "second", "third"]
        for i, msg in enumerate(messages):
            fifo.write(position=float(i), name=msg)
            fifo.finalize()
        
        for i, msg in enumerate(messages):
            data = fifo.read(timeout=0)
            assert abs(data.position.value - float(i)) < 1e-10
            assert data.name.value == msg
    
    def test_fifo_with_arrays(self, unique_name, cleanup_list):
        """Test FIFO with array fields."""
        @dataclass
        class ArrayData:
            data: "float32[5]" = None
        
        fifo = SharedMemory(ArrayData, name=unique_name, slots=3, create=True)
        cleanup_list.append(fifo)
        
        for i in range(3):
            arr = np.ones(5, dtype=np.float32) * i
            fifo.write(data=arr)
            fifo.finalize()
        
        for i in range(3):
            data = fifo.read(timeout=0)
            expected = np.ones(5, dtype=np.float32) * i
            np.testing.assert_array_almost_equal(data.data.value, expected)


# Manual test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])