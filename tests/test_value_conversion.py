"""
ValueWithStatus wrapper tests for flexible_shared_memory.

Tests cover:
- .value property access
- Status properties (valid, modified, truncated, unwritten)
- Magic method conversions (float, int, str, array)
- Arithmetic operations
- MULTIPROCESSING where SharedMemory is involved with auto-detection
"""

import pytest
import numpy as np
from dataclasses import dataclass
import multiprocessing
from multiprocessing import Queue
import sys

from flexible_shared_memory import SharedMemory, ValueWithStatus, FieldStatus


PROCESS_START_METHODS = ["fork", "spawn"] if sys.platform != "win32" else ["spawn"]


@dataclass
class ScalarData:
    temperature: float = 0.0
    count: int = 0
    active: bool = False


@dataclass
class StringData:
    message: "str[32]" = ""


@dataclass
class ArrayData:
    data: "float64[10]" = None


# Helper functions - ALL use ATTACH mode
def write_float(name: str, value: float, queue: Queue):
    """Write float value."""
    try:
        shm = SharedMemory(name)  # ATTACH
        shm.write(temperature=value)
        shm.close()
        queue.put(("success", value))
    except Exception as e:
        queue.put(("error", str(e)))


def write_int(name: str, value: int, queue: Queue):
    """Write int value."""
    try:
        shm = SharedMemory(name)  # ATTACH
        shm.write(count=value)
        shm.close()
        queue.put(("success", value))
    except Exception as e:
        queue.put(("error", str(e)))


def write_bool(name: str, value: bool, queue: Queue):
    """Write bool value."""
    try:
        shm = SharedMemory(name)  # ATTACH
        shm.write(active=value)
        shm.close()
        queue.put(("success", value))
    except Exception as e:
        queue.put(("error", str(e)))


def write_string(name: str, msg: str, queue: Queue):
    """Write string value."""
    try:
        shm = SharedMemory(name)  # ATTACH
        shm.write(message=msg)
        shm.close()
        queue.put(("success", msg))
    except Exception as e:
        queue.put(("error", str(e)))


def write_array(name: str, queue: Queue):
    """Write array value."""
    try:
        arr = np.arange(10, dtype=np.float64)
        shm = SharedMemory(name)  # ATTACH
        shm.write(data=arr)
        shm.close()
        queue.put(("success", arr.tolist()))
    except Exception as e:
        queue.put(("error", str(e)))


class TestValueProperty:
    """Test .value property access."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_value_property_float(self, start_method):
        """Test accessing float value via .value property."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(ScalarData)  # CREATE
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_float, args=(name, 23.5, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            temp_value = data.temperature.value
            assert isinstance(temp_value, float)
            assert abs(temp_value - 23.5) < 1e-10
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_value_property_int(self, start_method):
        """Test accessing int value via .value property."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(ScalarData)  # CREATE
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_int, args=(name, 42, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            count_value = data.count.value
            assert isinstance(count_value, int)
            assert count_value == 42
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_value_property_bool(self, start_method):
        """Test accessing bool value via .value property."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(ScalarData)  # CREATE
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_bool, args=(name, True, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            active_value = data.active.value
            assert isinstance(active_value, bool)
            assert active_value is True
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_value_property_string(self, start_method):
        """Test accessing string value via .value property."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(StringData)  # CREATE
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_string, args=(name, "Hello", queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            msg_value = data.message.value
            assert isinstance(msg_value, str)
            assert msg_value == "Hello"
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_value_property_array(self, start_method):
        """Test accessing array value via .value property."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(ArrayData)  # CREATE
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_array, args=(name, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            arr_value = data.data.value
            assert isinstance(arr_value, np.ndarray)
            expected = np.arange(10, dtype=np.float64)
            np.testing.assert_array_equal(arr_value, expected)
        finally:
            shm.close()
            shm.unlink()


class TestMagicConversions:
    """Test magic method conversions."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_float_conversion(self, start_method):
        """Test float() magic method."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(ScalarData)  # CREATE
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_float, args=(name, 23.5, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            temp = float(data.temperature)
            assert isinstance(temp, float)
            assert abs(temp - 23.5) < 1e-10
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_int_conversion(self, start_method):
        """Test int() magic method."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(ScalarData)  # CREATE
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_int, args=(name, 42, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            count = int(data.count)
            assert isinstance(count, int)
            assert count == 42
        finally:
            shm.close()
            shm.unlink()


class TestArithmeticOperations:
    """Test arithmetic operations on ValueWithStatus."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_addition(self, start_method):
        """Test addition operation."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(ScalarData)  # CREATE
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_float, args=(name, 20.0, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            result = data.temperature + 5.0
            assert abs(result - 25.0) < 1e-10
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_multiplication(self, start_method):
        """Test multiplication operation."""
        ctx = multiprocessing.get_context(start_method)
        shm = SharedMemory(ScalarData)  # CREATE
        name = shm.name
        
        try:
            queue = ctx.Queue()
            proc = ctx.Process(target=write_int, args=(name, 5, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            data = shm.read(timeout=0)
            result = data.count * 3
            assert result == 15
        finally:
            shm.close()
            shm.unlink()

class TestStatusFlags:
    """Test status flag behavior in detail."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_truncated_flag_string(self, start_method):
        """Test truncated flag is set when string is truncated."""
        
        @dataclass
        class ShortString:
            msg: "str[5]" = ""  # Only 5 chars max
        
        shm = SharedMemory(ShortString)  # CREATE
        
        try:
            # Write string longer than 5 chars
            long_string = "This is a very long message"
            shm.write(msg=long_string)
            
            data = shm.read(timeout=0)
            assert data is not None
            
            # Should be truncated
            assert data.msg.truncated, "String should be marked as truncated"
            assert not data.msg.valid, "Truncated data should not be valid"
            assert data.msg.modified, "Should be marked as modified"
            assert not data.msg.unwritten, "Should not be unwritten"
            
            # Value should be truncated to 5 chars
            assert len(data.msg.value) == 5, f"Expected 5 chars, got {len(data.msg.value)}"
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_truncated_flag_array(self, start_method):
        """Test truncated flag is set when array is truncated."""
        
        @dataclass
        class SmallArray:
            data: "float64[5]" = None  # Only 5 elements
        
        shm = SharedMemory(SmallArray)  # CREATE
        
        try:
            # Write array with 10 elements (more than 5)
            large_array = np.arange(10, dtype=np.float64)
            shm.write(data=large_array)
            
            data = shm.read(timeout=0)
            assert data is not None
            
            # Should be truncated
            assert data.data.truncated, "Array should be marked as truncated"
            assert not data.data.valid, "Truncated data should not be valid"
            
            # Should have only 5 elements
            assert len(data.data.value) == 5
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_unwritten_flag(self, start_method):
        """Test unwritten flag for fields that were never written."""
        shm = SharedMemory(ScalarData)  # CREATE
        
        try:
            # Write only temperature, leave count and active unwritten
            shm.write(temperature=25.0)
            
            data = shm.read(timeout=0)
            assert data is not None
            
            # Temperature was written
            assert not data.temperature.unwritten, "Written field should not be unwritten"
            assert data.temperature.valid, "Written field should be valid"
            assert data.temperature.modified, "Written field should be modified"
            
            # Count and active were NOT written
            assert data.count.unwritten, "Unwritten field should have unwritten flag"
            assert not data.count.valid, "Unwritten field should not be valid"
            assert not data.count.modified, "Unwritten field should not be modified"
            
            assert data.active.unwritten, "Unwritten field should have unwritten flag"
            assert not data.active.valid, "Unwritten field should not be valid"
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_overflow_flag_in_fifo(self, start_method):
        """Test overflow flag is set when FIFO overflows."""
        fifo = SharedMemory(ScalarData, slots=2)  # CREATE FIFO
        
        try:
            # Write 4 values to 2-slot FIFO (causes overflow)
            for i in range(4):
                fifo.write(temperature=float(i), count=i)
                fifo.finalize()
            
            # Read remaining values (should be 2, 3)
            data1 = fifo.read(timeout=1.0)
            assert data1 is not None
            assert data1.temperature.overflow, "Field should have overflow flag after FIFO overflow"
            assert data1.count.overflow, "All fields should have overflow flag"
            
            data2 = fifo.read(timeout=1.0)
            assert data2 is not None
            assert data2.temperature.overflow, "Second item should also have overflow flag"
        finally:
            fifo.close()
            fifo.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_valid_property_combines_flags(self, start_method):
        """Test that valid property is false when truncated OR unwritten."""
        
        @dataclass
        class MixedData:
            good: float = 0.0
            short: "str[3]" = ""
            unset: int = 0
        
        shm = SharedMemory(MixedData)  # CREATE
        
        try:
            # Write good field normally, short field with truncation, leave unset unwritten
            shm.write(good=42.0, short="toolong")
            
            data = shm.read(timeout=0)
            assert data is not None
            
            # good: valid (not truncated, not unwritten)
            assert data.good.valid, "Normal field should be valid"
            assert not data.good.truncated
            assert not data.good.unwritten
            
            # short: NOT valid (truncated)
            assert not data.short.valid, "Truncated field should not be valid"
            assert data.short.truncated
            assert not data.short.unwritten
            
            # unset: NOT valid (unwritten)
            assert not data.unset.valid, "Unwritten field should not be valid"
            assert not data.unset.truncated
            assert data.unset.unwritten
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_modified_flag_cleared_after_reset(self, start_method):
        """Test that modified flag is cleared by reset_modified=True."""
        shm = SharedMemory(ScalarData)  # CREATE
        
        try:
            # Write data
            shm.write(temperature=25.0, count=10)
            
            # Read without reset
            data1 = shm.read(timeout=0, reset_modified=False)
            assert data1 is not None
            assert data1.temperature.modified, "Should be modified after write"
            assert data1.count.modified, "Should be modified after write"
            
            # Read again without reset - still modified
            data2 = shm.read(timeout=0, reset_modified=False)
            assert data2 is not None
            assert data2.temperature.modified, "Should still be modified"
            
            # Read with reset
            data3 = shm.read(timeout=0, reset_modified=True)
            assert data3 is not None
            assert data3.temperature.modified, "Should be modified DURING this read"
            
            # Read again - now NOT modified
            data4 = shm.read(timeout=0, reset_modified=False)
            assert data4 is not None
            assert not data4.temperature.modified, "Should not be modified after reset"
            assert not data4.count.modified, "Should not be modified after reset"
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_flag_combinations(self, start_method):
        """Test various flag combinations work correctly."""
        
        @dataclass
        class FlagTest:
            normal: float = 0.0
            truncated_field: "str[2]" = ""
            unwritten_field: int = 0
        
        shm = SharedMemory(FlagTest)  # CREATE
        
        try:
            # First write: normal + truncated, leave unwritten
            shm.write(normal=1.0, truncated_field="abc")
            
            data = shm.read(timeout=0)
            
            # normal: valid=True, modified=True, truncated=False, unwritten=False
            assert data.normal.valid
            assert data.normal.modified
            assert not data.normal.truncated
            assert not data.normal.unwritten
            
            # truncated_field: valid=False, modified=True, truncated=True, unwritten=False
            assert not data.truncated_field.valid
            assert data.truncated_field.modified
            assert data.truncated_field.truncated
            assert not data.truncated_field.unwritten
            
            # unwritten_field: valid=False, modified=False, truncated=False, unwritten=True
            assert not data.unwritten_field.valid
            assert not data.unwritten_field.modified
            assert not data.unwritten_field.truncated
            assert data.unwritten_field.unwritten
        finally:
            shm.close()
            shm.unlink()


class TestValueWithStatusClass:
    """Test ValueWithStatus class directly (unit tests)."""
    
    def test_create_value_with_status(self):
        """Test creating ValueWithStatus instance."""
        status = FieldStatus(0b00000100)
        wrapper = ValueWithStatus(42.0, status)
        
        assert wrapper.value == 42.0
        assert wrapper.modified
        assert wrapper.valid
        assert not wrapper.truncated
        assert not wrapper.unwritten
    
    def test_with_different_types(self):
        """Test ValueWithStatus with various value types."""
        status = FieldStatus(0b00000100)
        
        wrapper_float = ValueWithStatus(1.5, status)
        assert float(wrapper_float) == 1.5
        
        wrapper_int = ValueWithStatus(42, status)
        assert int(wrapper_int) == 42
        
        wrapper_str = ValueWithStatus("test", status)
        assert str(wrapper_str) == "test"
        
        arr = np.array([1, 2, 3])
        wrapper_arr = ValueWithStatus(arr, status)
        np.testing.assert_array_equal(np.array(wrapper_arr), arr)

    def test_bool_conversion(self):
        """Test bool() magic method."""
        status = FieldStatus(0b00000100)
        
        # Truthy value
        wrapper_true = ValueWithStatus(42.0, status)
        assert bool(wrapper_true) is True
        
        # Falsy value
        wrapper_false = ValueWithStatus(0.0, status)
        assert bool(wrapper_false) is False
        
        wrapper_empty_str = ValueWithStatus("", status)
        assert bool(wrapper_empty_str) is False
    
    def test_repr(self):
        """Test __repr__ method."""
        status = FieldStatus(0b00000100)  # modified
        wrapper = ValueWithStatus(42.5, status)
        
        repr_str = repr(wrapper)
        assert "ValueWithStatus" in repr_str
        assert "42.5" in repr_str
        assert "valid=True" in repr_str
        assert "modified=True" in repr_str
    
    def test_subtraction(self):
        """Test subtraction operation."""
        status = FieldStatus(0b00000100)
        wrapper = ValueWithStatus(10.0, status)
        
        result = wrapper - 3.0
        assert result == 7.0
    
    def test_division(self):
        """Test division operation."""
        status = FieldStatus(0b00000100)
        wrapper = ValueWithStatus(20.0, status)
        
        result = wrapper / 4.0
        assert result == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
