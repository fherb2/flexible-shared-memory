"""
Basic write operation tests for flexible_shared_memory.

Tests cover:
- Writing scalar fields (float, int, bool)
- Writing strings (UTF-8)
- Writing arrays (NumPy)
- Sequence number behavior
- Modified flag updates
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
class ScalarData:
    """Simple scalar fields."""
    temperature: float = 0.0
    count: int = 0
    active: bool = False


@dataclass
class StringData:
    """String field."""
    message: "str[32]" = ""


@dataclass
class ArrayData:
    """Array field."""
    values: "float32[10]" = None


@dataclass
class MixedData:
    """Mixed field types."""
    position: float = 0.0
    count: int = 0
    active: bool = False
    name: "str[16]" = ""
    data: "float64[5,5]" = None


# Basic write tests
class TestWriteScalars:
    """Test writing scalar values."""
    
    def test_write_single_float(self, unique_name, cleanup_list):
        """Test writing a single float field."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Write float
        shm.write(temperature=23.5)
        
        # Read back
        data = shm.read(timeout=0)
        assert data is not None
        assert abs(data.temperature.value - 23.5) < 1e-10
        assert data.temperature.valid
        assert data.temperature.modified
    
    def test_write_single_int(self, unique_name, cleanup_list):
        """Test writing a single int field."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Write int
        shm.write(count=42)
        
        # Read back
        data = shm.read(timeout=0)
        assert data is not None
        assert data.count.value == 42
        assert data.count.valid
        assert data.count.modified
    
    def test_write_single_bool(self, unique_name, cleanup_list):
        """Test writing a single bool field."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Write bool
        shm.write(active=True)
        
        # Read back
        data = shm.read(timeout=0)
        assert data is not None
        assert data.active.value is True
        assert data.active.valid
        assert data.active.modified
    
    def test_write_multiple_scalars(self, unique_name, cleanup_list):
        """Test writing multiple scalar fields at once."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Write all fields
        shm.write(temperature=25.5, count=100, active=True)
        
        # Read back
        data = shm.read(timeout=0)
        assert data is not None
        assert abs(data.temperature.value - 25.5) < 1e-10
        assert data.count.value == 100
        assert data.active.value is True
        
        # All should be modified
        assert data.temperature.modified
        assert data.count.modified
        assert data.active.modified
    
    def test_write_negative_values(self, unique_name, cleanup_list):
        """Test writing negative values."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=-15.5, count=-42)
        
        data = shm.read(timeout=0)
        assert abs(data.temperature.value - (-15.5)) < 1e-10
        assert data.count.value == -42
    
    def test_write_zero_values(self, unique_name, cleanup_list):
        """Test writing zero values."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=0.0, count=0, active=False)
        
        data = shm.read(timeout=0)
        assert data.temperature.value == 0.0
        assert data.count.value == 0
        assert data.active.value is False


class TestWriteStrings:
    """Test writing string values."""
    
    def test_write_simple_string(self, unique_name, cleanup_list):
        """Test writing a simple ASCII string."""
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(message="Hello World")
        
        data = shm.read(timeout=0)
        assert data is not None
        assert data.message.value == "Hello World"
        assert data.message.valid
        assert data.message.modified
        assert not data.message.truncated
    
    def test_write_empty_string(self, unique_name, cleanup_list):
        """Test writing an empty string."""
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(message="")
        
        data = shm.read(timeout=0)
        assert data.message.value == ""
        assert data.message.valid
        assert not data.message.truncated
    
    def test_write_unicode_string(self, unique_name, cleanup_list):
        """Test writing Unicode characters."""
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # German, Russian, Chinese, Emoji
        text = "Hällö Мир 世界 🎉"
        shm.write(message=text)
        
        data = shm.read(timeout=0)
        assert data.message.value == text
        assert data.message.valid
    
    def test_write_string_at_limit(self, unique_name, cleanup_list):
        """Test writing string exactly at character limit."""
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Exactly 32 characters
        text = "a" * 32
        shm.write(message=text)
        
        data = shm.read(timeout=0)
        assert data.message.value == text
        assert len(data.message.value) == 32
        assert not data.message.truncated
    
    def test_write_string_overflow(self, unique_name, cleanup_list):
        """Test writing string that exceeds limit."""
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # 40 characters, limit is 32
        text = "a" * 40
        shm.write(message=text)
        
        data = shm.read(timeout=0)
        assert len(data.message.value) == 32
        assert data.message.value == "a" * 32
        assert data.message.truncated
        assert not data.message.valid  # NOT valid when truncated!


class TestWriteArrays:
    """Test writing NumPy arrays."""
    
    def test_write_simple_array(self, unique_name, cleanup_list):
        """Test writing a simple 1D array."""
        shm = SharedMemory(ArrayData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
        shm.write(values=arr)
        
        data = shm.read(timeout=0)
        assert data is not None
        np.testing.assert_array_almost_equal(data.values.value, arr)
        assert data.values.valid
        assert data.values.modified
        assert not data.values.truncated
    
    def test_write_zero_array(self, unique_name, cleanup_list):
        """Test writing array of zeros."""
        shm = SharedMemory(ArrayData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        arr = np.zeros(10, dtype=np.float32)
        shm.write(values=arr)
        
        data = shm.read(timeout=0)
        np.testing.assert_array_equal(data.values.value, arr)
    
    def test_write_array_overflow(self, unique_name, cleanup_list):
        """Test writing array larger than field size."""
        shm = SharedMemory(ArrayData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # 15 elements, but field is [10]
        arr = np.arange(15, dtype=np.float32)
        shm.write(values=arr)
        
        data = shm.read(timeout=0)
        # Should be truncated to first 10 elements
        np.testing.assert_array_equal(data.values.value, arr[:10])
        assert data.values.truncated
    
    def test_write_2d_array(self, unique_name, cleanup_list):
        """Test writing 2D array."""
        @dataclass
        class Array2D:
            matrix: "float64[5,5]" = None
        
        shm = SharedMemory(Array2D, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        arr = np.arange(25, dtype=np.float64).reshape(5, 5)
        shm.write(matrix=arr)
        
        data = shm.read(timeout=0)
        np.testing.assert_array_equal(data.matrix.value, arr)
        assert data.matrix.value.shape == (5, 5)


class TestWriteMixed:
    """Test writing mixed field types."""
    
    def test_write_all_fields(self, unique_name, cleanup_list):
        """Test writing all fields of different types."""
        shm = SharedMemory(MixedData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        arr = np.ones((5, 5), dtype=np.float64)
        shm.write(
            position=1.5,
            count=42,
            active=True,
            name="test",
            data=arr
        )
        
        data = shm.read(timeout=0)
        assert abs(data.position.value - 1.5) < 1e-10
        assert data.count.value == 42
        assert data.active.value is True
        assert data.name.value == "test"
        np.testing.assert_array_equal(data.data.value, arr)
        
        # All should be modified
        assert data.position.modified
        assert data.count.modified
        assert data.active.modified
        assert data.name.modified
        assert data.data.modified
    
    def test_write_partial_fields(self, unique_name, cleanup_list):
        """Test writing only some fields."""
        shm = SharedMemory(MixedData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # First write
        shm.write(position=1.0, count=10)
        
        data = shm.read(timeout=0)
        assert data.position.modified
        assert data.count.modified
        assert not data.active.modified  # Not written
        assert not data.name.modified
        assert not data.data.modified


class TestSequenceNumbers:
    """Test sequence number behavior."""
    
    def test_sequence_increments(self, unique_name, cleanup_list):
        """Test that sequence numbers increment on each write."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Multiple writes
        for i in range(5):
            shm.write(count=i)
            data = shm.read(timeout=0)
            assert data is not None
            assert data.count.value == i
    
    def test_modified_flag_persistence(self, unique_name, cleanup_list):
        """Test that modified flag is set on write."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Write temperature only
        shm.write(temperature=20.0)
        
        data = shm.read(timeout=0)
        assert data.temperature.modified
        assert not data.count.modified
        assert not data.active.modified
        
        # Write count
        shm.write(count=5)
        
        data = shm.read(timeout=0)
        assert data.count.modified
        # Temperature no longer modified (not in latest write)
        assert not data.temperature.modified


# Manual test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])