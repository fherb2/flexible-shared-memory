"""
Field status flag tests for flexible_shared_memory.

Tests cover:
- valid flag
- modified flag
- truncated flag
- unwritten flag
- Flag persistence and behavior
- ALL with multiprocessing (fork and spawn) and auto-detection
"""

import pytest
import numpy as np
from dataclasses import dataclass
import multiprocessing
from multiprocessing import Queue
import sys

from flexible_shared_memory import SharedMemory, FieldStatus


PROCESS_START_METHODS = ["fork", "spawn"] if sys.platform != "win32" else ["spawn"]


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


# Helper functions - ALL use ATTACH mode
def write_both_fields(name: str, value: float, count: int, queue: Queue):
    """Write both fields."""
    try:
        shm = SharedMemory(name)  # ATTACH
        shm.write(value=value, count=count)
        shm.close()
        queue.put(("success", None))
    except Exception as e:
        queue.put(("error", str(e)))


def write_single_field(name: str, value: float, queue: Queue):
    """Write only value field."""
    try:
        shm = SharedMemory(name)  # ATTACH
        shm.write(value=value)
        shm.close()
        queue.put(("success", None))
    except Exception as e:
        queue.put(("error", str(e)))


def write_long_string(name: str, length: int, queue: Queue):
    """Write string longer than limit."""
    try:
        shm = SharedMemory(name)  # ATTACH
        shm.write(message="a" * length)
        shm.close()
        queue.put(("success", None))
    except Exception as e:
        queue.put(("error", str(e)))


def write_oversized_array(name: str, queue: Queue):
    """Write array larger than field."""
    try:
        shm = SharedMemory(name)  # ATTACH
        arr = np.arange(15, dtype=np.float32)
        shm.write(data=arr)
        shm.close()
        queue.put(("success", None))
    except Exception as e:
        queue.put(("error", str(e)))


def write_mixed_truncated(name: str, queue: Queue):
    """Write mixed data with truncated string."""
    try:
        shm = SharedMemory(name)  # ATTACH
        shm.write(
            position=1.0,
            name="a" * 30,
            values=np.arange(5, dtype=np.int32)
        )
        shm.close()
        queue.put(("success", None))
    except Exception as e:
        queue.put(("error", str(e)))


class TestValidFlag:
    """Test the valid flag behavior."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_valid_after_write(self, start_method):
        """Test that written fields are marked as valid."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(SimpleData)  # CREATE with auto-generated name
        name = shm.name  # Get generated name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_both_fields, args=(name, 1.0, 1, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            assert data.value.valid
            assert data.count.valid
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_unwritten_not_valid(self, start_method):
        """Test that unwritten fields are not valid."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(SimpleData)
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_single_field, args=(name, 1.0, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            assert data.value.valid
            assert not data.count.valid
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_truncated_not_valid(self, start_method):
        """Test that truncated fields are NOT valid."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(StringData)
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_long_string, args=(name, 50, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            assert not data.message.valid
            assert data.message.truncated
        finally:
            shm.close()
            shm.unlink()


class TestUnwrittenFlag:
    """Test the unwritten flag behavior."""
    
    def test_unwritten_on_new_slot(self):
        """Test that new slots have unwritten flags set."""
        shm = SharedMemory(SimpleData)
        
        try:
            data = shm.read(timeout=0)
            if data is not None:
                assert data.value.unwritten
                assert data.count.unwritten
        finally:
            shm.close()
            shm.unlink()


class TestModifiedFlag:
    """Test the modified flag behavior."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_modified_set_on_write(self, start_method):
        """Test that modified flag is set when field is written."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(SimpleData)
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_both_fields, args=(name, 1.0, 1, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            assert data.value.modified
            assert data.count.modified
        finally:
            shm.close()
            shm.unlink()

    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_modified_cleared_by_reset(self, start_method):
        """Test that modified flag is cleared by reset_modified=True."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(SimpleData)
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_both_fields, args=(name, 1.0, 1, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Read without reset - should be modified
            data1 = shm.read(timeout=0, reset_modified=False)
            assert data1.value.modified, "Should be modified after write"
            assert data1.count.modified, "Should be modified after write"
            
            # Read again without reset - still modified
            data2 = shm.read(timeout=0, reset_modified=False)
            assert data2.value.modified, "Should still be modified"
            
            # Read with reset - modified during this read
            data3 = shm.read(timeout=0, reset_modified=True)
            assert data3.value.modified, "Should be modified during reset read"
            
            # Read again - now NOT modified
            data4 = shm.read(timeout=0, reset_modified=False)
            assert not data4.value.modified, "Should not be modified after reset"
            assert not data4.count.modified, "Should not be modified after reset"
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_modified_not_set_for_unwritten_fields(self, start_method):
        """Test that modified flag is not set for fields that weren't written."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(SimpleData)
        name = shm.name
        
        try:
            # Write only value, not count
            queue = ctx.Queue()
            proc = ctx.Process(target=write_single_field, args=(name, 1.0, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            
            # value was written - should be modified
            assert data.value.modified, "Written field should be modified"
            
            # count was NOT written - should NOT be modified
            assert not data.count.modified, "Unwritten field should not be modified"
        finally:
            shm.close()
            shm.unlink()


class TestTruncatedFlag:
    """Test the truncated flag behavior."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_truncated_for_long_string(self, start_method):
        """Test that truncated flag is set for oversized strings."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(StringData)
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_long_string, args=(name, 40, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            assert data.message.truncated
            assert len(data.message.value) == 32
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_truncated_for_oversized_array(self, start_method):
        """Test that truncated flag is set for oversized arrays."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(ArrayData)  # CREATE
        name = shm.name  # Get generated name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_oversized_array, args=(name, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            assert data.data.truncated
            assert len(data.data.value) == 10
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_truncated_only_affects_field(self, start_method):
        """Test that truncated flag only affects the truncated field."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(MixedData)  # CREATE
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_mixed_truncated, args=(name, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            assert not data.position.truncated
            assert data.name.truncated
            assert not data.values.truncated
        finally:
            shm.close()
            shm.unlink()


class TestFieldStatusClass:
    """Test FieldStatus class directly."""
    
    def test_field_status_properties(self):
        """Test FieldStatus property access."""
        status = FieldStatus(0b00000111)
        assert status.is_truncated
        assert status.is_unwritten
        assert status.is_modified
        assert not status.is_valid
        
        status = FieldStatus(0b00000100)
        assert not status.is_truncated
        assert not status.is_unwritten
        assert status.is_modified
        assert status.is_valid
    
    def test_field_status_bit_masks(self):
        """Test FieldStatus bit mask constants."""
        assert FieldStatus.MASK_TRUNCATED == 0b00000001
        assert FieldStatus.MASK_UNWRITTEN == 0b00000010
        assert FieldStatus.MASK_MODIFIED == 0b00000100
        assert FieldStatus.MASK_OVERFLOW == 0b00001000
    
    def test_field_status_overflow_property(self):
        """Test FieldStatus overflow property."""
        # No overflow
        status = FieldStatus(0b00000100)  # Only modified
        assert not status.is_overflow
        assert status.is_valid
        
        # With overflow
        status = FieldStatus(0b00001100)  # Modified + overflow
        assert status.is_overflow
        assert status.is_modified
        
        # Overflow affects valid
        status = FieldStatus(0b00001000)  # Only overflow
        assert status.is_overflow
        # Note: overflow doesn't affect valid (only truncated/unwritten do)
        assert status.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
