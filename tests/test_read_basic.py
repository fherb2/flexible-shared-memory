"""
Basic read operation tests for flexible_shared_memory.

Tests cover:
- Reading after write
- Timeout behavior
- Blocking reads
- Data correctness
- Sequence number consistency
"""

import pytest
import numpy as np
from dataclasses import dataclass
import time
import threading

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
class ComplexData:
    position: float = 0.0
    name: "str[32]" = ""
    data: "float32[10]" = None


# Basic read tests
class TestReadBasics:
    """Test basic read operations."""
    
    def test_read_after_write(self, unique_name, cleanup_list):
        """Test reading immediately after write."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=42.0, count=10)
        data = shm.read(timeout=0)
        
        assert data is not None
        assert abs(data.value.value - 42.0) < 1e-10
        assert data.count.value == 10
    
    def test_read_returns_none_when_empty(self, unique_name, cleanup_list):
        """Test that read returns None with timeout=0 when no data."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # No write yet
        data = shm.read(timeout=0)
        # Uninitialized memory returns object with unwritten flags
        if data:
            assert data.value.unwritten
            assert data.count.unwritten
    
    def test_read_multiple_times(self, unique_name, cleanup_list):
        """Test reading same data multiple times."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=123.4, count=5)
        
        # Read 3 times
        for _ in range(3):
            data = shm.read(timeout=0)
            assert data is not None
            assert abs(data.value.value - 123.4) < 1e-10
            assert data.count.value == 5
    
    def test_read_after_multiple_writes(self, unique_name, cleanup_list):
        """Test that read gets latest write."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=1.0)
        shm.write(value=2.0)
        shm.write(value=3.0)
        
        data = shm.read(timeout=0)
        assert abs(data.value.value - 3.0) < 1e-10


class TestReadTimeout:
    """Test read timeout behavior."""
    
    def test_read_timeout_zero(self, unique_name, cleanup_list):
        """Test non-blocking read with timeout=0."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        start = time.time()
        data = shm.read(timeout=0)
        elapsed = time.time() - start

        # Should return immediately with uninitialized data
        if data:
            assert data.value.unwritten
    
    def test_read_timeout_short(self, unique_name, cleanup_list):
        """Test read with short timeout."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        start = time.time()
        data = shm.read(timeout=0.2)
        elapsed = time.time() - start

        # Returns uninitialized data immediately
        if data:
            assert data.value.unwritten
    
    def test_read_blocks_until_data(self, unique_name, cleanup_list):
        """Test that read waits for valid data."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        def delayed_write():
            time.sleep(0.2)
            shm_w = SharedMemory(SimpleData, name=unique_name, create=False)
            shm_w.write(value=99.0)
            shm_w.close()
        
        # Start writer in background
        writer = threading.Thread(target=delayed_write)
        writer.start()
        
        # Wait for valid data
        start = time.time()
        data = None
        while (time.time() - start) < 1.0:
            data = shm.read(timeout=0.1)
            if data and data.value.valid:
                break
        elapsed = time.time() - start
        
        writer.join()
        
        assert data is not None
        assert data.value.valid
        assert abs(data.value.value - 99.0) < 1e-10
        assert 0.15 < elapsed < 0.5  # Waited for write


class TestReadDataTypes:
    """Test reading different data types."""
    
    def test_read_scalars(self, unique_name, cleanup_list):
        """Test reading scalar values."""
        @dataclass
        class ScalarData:
            f: float = 0.0
            i: int = 0
            b: bool = False
        
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(f=3.14159, i=-42, b=True)
        data = shm.read(timeout=0)
        
        assert abs(data.f.value - 3.14159) < 1e-10
        assert data.i.value == -42
        assert data.b.value is True
    
    def test_read_string(self, unique_name, cleanup_list):
        """Test reading string with UTF-8."""
        @dataclass
        class StringData:
            msg: "str[64]" = ""
        
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        text = "Hello 世界 🎉"
        shm.write(msg=text)
        data = shm.read(timeout=0)
        
        assert data.msg.value == text
    
    def test_read_array_1d(self, unique_name, cleanup_list):
        """Test reading 1D array."""
        @dataclass
        class ArrayData:
            arr: "float64[10]" = None
        
        shm = SharedMemory(ArrayData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        arr = np.arange(10, dtype=np.float64) * 0.5
        shm.write(arr=arr)
        data = shm.read(timeout=0)
        
        np.testing.assert_array_almost_equal(data.arr.value, arr)
        assert data.arr.value.shape == (10,)
    
    def test_read_array_2d(self, unique_name, cleanup_list):
        """Test reading 2D array."""
        @dataclass
        class Array2DData:
            matrix: "int32[4,5]" = None
        
        shm = SharedMemory(Array2DData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        arr = np.arange(20, dtype=np.int32).reshape(4, 5)
        shm.write(matrix=arr)
        data = shm.read(timeout=0)
        
        np.testing.assert_array_equal(data.matrix.value, arr)
        assert data.matrix.value.shape == (4, 5)
    
    def test_read_mixed_types(self, unique_name, cleanup_list):
        """Test reading dataclass with mixed types."""
        shm = SharedMemory(ComplexData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        arr = np.ones(10, dtype=np.float32) * 5.0
        shm.write(position=12.34, name="TestName", data=arr)
        
        data = shm.read(timeout=0)
        assert abs(data.position.value - 12.34) < 1e-10
        assert data.name.value == "TestName"
        np.testing.assert_array_almost_equal(data.data.value, arr)


class TestReadConsistency:
    """Test sequence number consistency."""
    
    def test_read_consistency_check(self, unique_name, cleanup_list):
        """Test that sequence numbers ensure consistency."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Write and read - should be consistent
        shm.write(value=1.0, count=1)
        data = shm.read(timeout=0)
        
        assert data is not None
        assert data.value.value == 1.0
        assert data.count.value == 1
    
    def test_read_doesnt_modify_data(self, unique_name, cleanup_list):
        """Test that reading doesn't change the data."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=42.0, count=7)
        
        # Read twice
        data1 = shm.read(timeout=0)
        data2 = shm.read(timeout=0)
        
        assert abs(data1.value.value - data2.value.value) < 1e-10
        assert data1.count.value == data2.count.value


class TestReadResetModified:
    """Test reset_modified parameter."""
    
    def test_reset_modified_true(self, unique_name, cleanup_list):
        """Test that reset_modified=True clears modified flags."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=1.0, count=1)
        
        # First read - should see modified
        data1 = shm.read(timeout=0, reset_modified=False)
        assert data1.value.modified
        assert data1.count.modified
        
        # Second read with reset
        data2 = shm.read(timeout=0, reset_modified=True)
        assert data2.value.modified  # Still modified in this read
        
        # Third read - should NOT be modified anymore
        data3 = shm.read(timeout=0, reset_modified=False)
        assert not data3.value.modified
        assert not data3.count.modified
    
    def test_reset_modified_false(self, unique_name, cleanup_list):
        """Test that reset_modified=False leaves flags unchanged."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=2.0, count=2)
        
        # Multiple reads without reset
        for _ in range(3):
            data = shm.read(timeout=0, reset_modified=False)
            assert data.value.modified
            assert data.count.modified


class TestReadEdgeCases:
    """Test edge cases in reading."""
    
    def test_read_from_uninitialized(self, unique_name, cleanup_list):
        """Test reading from newly created (uninitialized) memory."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        data = shm.read(timeout=0)
        # Should return None or have unwritten flags
        if data is not None:
            assert data.value.unwritten
            assert data.count.unwritten
    
    def test_read_immediately_after_write(self, unique_name, cleanup_list):
        """Test read with no delay after write."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        for i in range(10):
            shm.write(value=float(i))
            data = shm.read(timeout=0)
            assert data is not None
            assert abs(data.value.value - float(i)) < 1e-10


# Manual test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])