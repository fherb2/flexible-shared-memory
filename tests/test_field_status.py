"""
Field status flag tests for flexible_shared_memory.

Tests cover:
- valid flag
- modified flag
- truncated flag
- unwritten flag
- Flag persistence and behavior
"""

import pytest
import numpy as np
from dataclasses import dataclass
import time

from flexible_shared_memory import SharedMemory, FieldStatus


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
class StringData:
    message: "str[32]" = ""


@dataclass
class ArrayData:
    data: "float32[10]" = None


@dataclass
class MixedData:
    position: float = 0.0
    name: "str[16]" = ""
    values: "int32[5]" = None


# Valid flag tests
class TestValidFlag:
    """Test the valid flag behavior."""
    
    def test_valid_after_write(self, unique_name, cleanup_list):
        """Test that written fields are marked as valid."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=1.0, count=1)
        data = shm.read(timeout=0)
        
        assert data.value.valid
        assert data.count.valid
    
    def test_unwritten_not_valid(self, unique_name, cleanup_list):
        """Test that unwritten fields are not valid."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Write only one field
        shm.write(value=1.0)
        data = shm.read(timeout=0)
        
        assert data.value.valid
        assert not data.count.valid  # Unwritten
    
    def test_truncated_not_valid(self, unique_name, cleanup_list):
        """Test that truncated fields are NOT valid."""
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # String too long
        shm.write(message="a" * 50)  # Limit is 32
        data = shm.read(timeout=0)
        
        assert not data.message.valid  # Truncated = NOT valid
        assert data.message.truncated


# Unwritten flag tests
class TestUnwrittenFlag:
    """Test the unwritten flag behavior."""
    
    def test_unwritten_on_new_slot(self, unique_name, cleanup_list):
        """Test that new slots have unwritten flags set."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        data = shm.read(timeout=0)
        if data is not None:
            assert data.value.unwritten
            assert data.count.unwritten
    
    def test_unwritten_cleared_after_write(self, unique_name, cleanup_list):
        """Test that unwritten flag is cleared after first write."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=1.0)
        data = shm.read(timeout=0)
        
        assert not data.value.unwritten
        assert data.count.unwritten  # Still unwritten
        
        # Write the other field
        shm.write(count=5)
        data = shm.read(timeout=0)
        
        assert not data.count.unwritten  # Now written
    
    def test_unwritten_stays_cleared(self, unique_name, cleanup_list):
        """Test that unwritten flag stays cleared after multiple writes."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=1.0)
        
        # Write same field multiple times
        for i in range(5):
            shm.write(value=float(i))
            data = shm.read(timeout=0)
            assert not data.value.unwritten


# Modified flag tests
class TestModifiedFlag:
    """Test the modified flag behavior."""
    
    def test_modified_set_on_write(self, unique_name, cleanup_list):
        """Test that modified flag is set when field is written."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=1.0, count=1)
        data = shm.read(timeout=0)
        
        assert data.value.modified
        assert data.count.modified
    
    def test_modified_only_for_written_fields(self, unique_name, cleanup_list):
        """Test that modified is only set for fields in current write."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # First write
        shm.write(value=1.0)
        data = shm.read(timeout=0)
        assert data.value.modified
        assert not data.count.modified
        
        # Second write - different field
        shm.write(count=5)
        data = shm.read(timeout=0)
        assert not data.value.modified  # Not in this write
        assert data.count.modified
    
    def test_modified_reset_with_flag(self, unique_name, cleanup_list):
        """Test that modified can be reset with reset_modified=True."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=1.0)
        
        # Read with reset
        data = shm.read(timeout=0, reset_modified=True)
        assert data.value.modified  # Still modified in this read
        
        # Next read - should be clear
        data = shm.read(timeout=0)
        assert not data.value.modified
    
    def test_modified_persists_across_reads(self, unique_name, cleanup_list):
        """Test that modified persists without reset."""
        shm = SharedMemory(SimpleData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(value=1.0)
        
        # Multiple reads without reset
        for _ in range(3):
            data = shm.read(timeout=0, reset_modified=False)
            assert data.value.modified


# Truncated flag tests
class TestTruncatedFlag:
    """Test the truncated flag behavior."""
    
    def test_truncated_for_long_string(self, unique_name, cleanup_list):
        """Test that truncated flag is set for oversized strings."""
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # String longer than 32 chars
        shm.write(message="a" * 40)
        data = shm.read(timeout=0)
        
        assert data.message.truncated
        assert len(data.message.value) == 32
    
    def test_not_truncated_at_limit(self, unique_name, cleanup_list):
        """Test that truncated is not set when exactly at limit."""
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Exactly 32 chars
        shm.write(message="a" * 32)
        data = shm.read(timeout=0)
        
        assert not data.message.truncated
        assert data.message.valid
    
    def test_truncated_for_oversized_array(self, unique_name, cleanup_list):
        """Test that truncated flag is set for oversized arrays."""
        shm = SharedMemory(ArrayData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Array larger than [10]
        arr = np.arange(15, dtype=np.float32)
        shm.write(data=arr)
        data = shm.read(timeout=0)
        
        assert data.data.truncated
        assert len(data.data.value) == 10
    
    def test_not_truncated_for_correct_array(self, unique_name, cleanup_list):
        """Test that truncated is not set for correct array size."""
        shm = SharedMemory(ArrayData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        arr = np.arange(10, dtype=np.float32)
        shm.write(data=arr)
        data = shm.read(timeout=0)
        
        assert not data.data.truncated
        assert data.data.valid
    
    def test_truncated_only_affects_field(self, unique_name, cleanup_list):
        """Test that truncated flag only affects the truncated field."""
        shm = SharedMemory(MixedData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Truncate only the string
        shm.write(
            position=1.0,
            name="a" * 30,  # Limit is 16
            values=np.arange(5, dtype=np.int32)
        )
        data = shm.read(timeout=0)
        
        assert not data.position.truncated
        assert data.name.truncated
        assert not data.values.truncated


# FieldStatus class tests
class TestFieldStatusClass:
    """Test FieldStatus class directly."""
    
    def test_field_status_properties(self):
        """Test FieldStatus property access."""
        # All flags set
        status = FieldStatus(0b00000111)
        assert status.is_truncated
        assert status.is_unwritten
        assert status.is_modified
        assert not status.is_valid  # Not valid if truncated or unwritten
        
        # Only modified
        status = FieldStatus(0b00000100)
        assert not status.is_truncated
        assert not status.is_unwritten
        assert status.is_modified
        assert status.is_valid
        
        # None set (initial state after write without issues)
        status = FieldStatus(0b00000000)
        assert not status.is_truncated
        assert not status.is_unwritten
        assert not status.is_modified
        assert status.is_valid
    
    def test_field_status_bit_masks(self):
        """Test FieldStatus bit mask constants."""
        assert FieldStatus.MASK_TRUNCATED == 0b00000001
        assert FieldStatus.MASK_UNWRITTEN == 0b00000010
        assert FieldStatus.MASK_MODIFIED == 0b00000100


# Combined flag tests
class TestCombinedFlags:
    """Test multiple flags simultaneously."""
    
    def test_modified_and_truncated(self, unique_name, cleanup_list):
        """Test field that is both modified and truncated."""
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(message="a" * 50)
        data = shm.read(timeout=0)
        
        assert data.message.modified
        assert data.message.truncated
        assert not data.message.valid  # NOT valid when truncated
        assert not data.message.unwritten
    
    def test_independent_field_status(self, unique_name, cleanup_list):
        """Test that each field has independent status."""
        shm = SharedMemory(MixedData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        # Write with one field truncated
        shm.write(
            position=1.0,
            name="a" * 20,  # Truncated
            values=np.arange(5, dtype=np.int32)
        )
        data = shm.read(timeout=0)
        
        # Check each field independently
        assert data.position.valid
        assert data.position.modified
        assert not data.position.truncated
        
        assert not data.name.valid  # NOT valid when truncated
        assert data.name.modified
        assert data.name.truncated
        
        assert data.values.valid
        assert data.values.modified
        assert not data.values.truncated


# Manual test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])