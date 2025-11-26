"""
ValueWithStatus wrapper tests for flexible_shared_memory.

Tests cover:
- .value property access
- Status properties (valid, modified, truncated, unwritten)
- Magic method conversions (float, int, str, array)
- Arithmetic operations
"""

import pytest
import numpy as np
from dataclasses import dataclass
import time

from flexible_shared_memory import SharedMemory, ValueWithStatus, FieldStatus


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
    temperature: float = 0.0
    count: int = 0
    active: bool = False


@dataclass
class StringData:
    message: "str[32]" = ""


@dataclass
class ArrayData:
    data: "float64[10]" = None


# Value property tests
class TestValueProperty:
    """Test .value property access."""
    
    def test_value_property_float(self, unique_name, cleanup_list):
        """Test accessing float value via .value property."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=23.5)
        data = shm.read(timeout=0)
        
        # Access via .value
        temp_value = data.temperature.value
        assert isinstance(temp_value, float)
        assert abs(temp_value - 23.5) < 1e-10
    
    def test_value_property_int(self, unique_name, cleanup_list):
        """Test accessing int value via .value property."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(count=42)
        data = shm.read(timeout=0)
        
        count_value = data.count.value
        assert isinstance(count_value, int)
        assert count_value == 42
    
    def test_value_property_bool(self, unique_name, cleanup_list):
        """Test accessing bool value via .value property."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(active=True)
        data = shm.read(timeout=0)
        
        active_value = data.active.value
        assert isinstance(active_value, bool)
        assert active_value is True
    
    def test_value_property_string(self, unique_name, cleanup_list):
        """Test accessing string value via .value property."""
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(message="Hello World")
        data = shm.read(timeout=0)
        
        msg_value = data.message.value
        assert isinstance(msg_value, str)
        assert msg_value == "Hello World"
    
    def test_value_property_array(self, unique_name, cleanup_list):
        """Test accessing array value via .value property."""
        shm = SharedMemory(ArrayData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        arr = np.arange(10, dtype=np.float64)
        shm.write(data=arr)
        data = shm.read(timeout=0)
        
        arr_value = data.data.value
        assert isinstance(arr_value, np.ndarray)
        np.testing.assert_array_equal(arr_value, arr)


# Status property tests
class TestStatusProperties:
    """Test status properties on ValueWithStatus."""
    
    def test_valid_property(self, unique_name, cleanup_list):
        """Test .valid property access."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=1.0)
        data = shm.read(timeout=0)
        
        assert data.temperature.valid is True
        assert data.count.valid is False  # Unwritten
    
    def test_modified_property(self, unique_name, cleanup_list):
        """Test .modified property access."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=1.0)
        data = shm.read(timeout=0)
        
        assert data.temperature.modified is True
        assert data.count.modified is False
    
    def test_truncated_property(self, unique_name, cleanup_list):
        """Test .truncated property access."""
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(message="a" * 50)  # Truncated
        data = shm.read(timeout=0)
        
        assert data.message.truncated is True
    
    def test_unwritten_property(self, unique_name, cleanup_list):
        """Test .unwritten property access."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        data = shm.read(timeout=0)
        if data is not None:
            assert data.temperature.unwritten is True


# Magic method conversion tests
class TestMagicConversions:
    """Test magic method conversions."""
    
    def test_float_conversion(self, unique_name, cleanup_list):
        """Test float() magic method."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=23.5)
        data = shm.read(timeout=0)
        
        # Convert using float()
        temp = float(data.temperature)
        assert isinstance(temp, float)
        assert abs(temp - 23.5) < 1e-10
    
    def test_int_conversion(self, unique_name, cleanup_list):
        """Test int() magic method."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(count=42)
        data = shm.read(timeout=0)
        
        # Convert using int()
        count = int(data.count)
        assert isinstance(count, int)
        assert count == 42
    
    def test_bool_conversion(self, unique_name, cleanup_list):
        """Test bool() magic method."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(active=True)
        data = shm.read(timeout=0)
        
        # Convert using bool()
        active = bool(data.active)
        assert isinstance(active, bool)
        assert active is True
    
    def test_str_conversion(self, unique_name, cleanup_list):
        """Test str() magic method."""
        shm = SharedMemory(StringData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(message="Test")
        data = shm.read(timeout=0)
        
        # Convert using str()
        msg = str(data.message)
        assert isinstance(msg, str)
        assert msg == "Test"
    
    def test_array_conversion(self, unique_name, cleanup_list):
        """Test np.array() magic method."""
        shm = SharedMemory(ArrayData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        arr = np.arange(10, dtype=np.float64)
        shm.write(data=arr)
        data = shm.read(timeout=0)
        
        # Convert using np.array()
        arr_copy = np.array(data.data)
        assert isinstance(arr_copy, np.ndarray)
        np.testing.assert_array_equal(arr_copy, arr)


# Arithmetic operation tests
class TestArithmeticOperations:
    """Test arithmetic operations on ValueWithStatus."""
    
    def test_addition(self, unique_name, cleanup_list):
        """Test addition operation."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=20.0)
        data = shm.read(timeout=0)
        
        result = data.temperature + 5.0
        assert abs(result - 25.0) < 1e-10
    
    def test_subtraction(self, unique_name, cleanup_list):
        """Test subtraction operation."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=30.0)
        data = shm.read(timeout=0)
        
        result = data.temperature - 10.0
        assert abs(result - 20.0) < 1e-10
    
    def test_multiplication(self, unique_name, cleanup_list):
        """Test multiplication operation."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(count=5)
        data = shm.read(timeout=0)
        
        result = data.count * 3
        assert result == 15
    
    def test_division(self, unique_name, cleanup_list):
        """Test division operation."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=100.0)
        data = shm.read(timeout=0)
        
        result = data.temperature / 4.0
        assert abs(result - 25.0) < 1e-10
    
    def test_chained_operations(self, unique_name, cleanup_list):
        """Test chained arithmetic operations."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=10.0)
        data = shm.read(timeout=0)
        
        result = (data.temperature + 5) * 2 - 10
        assert abs(result - 20.0) < 1e-10


# ValueWithStatus class tests
class TestValueWithStatusClass:
    """Test ValueWithStatus class directly."""
    
    def test_create_value_with_status(self):
        """Test creating ValueWithStatus instance."""
        status = FieldStatus(0b00000100)  # Only modified
        wrapper = ValueWithStatus(42.0, status)
        
        assert wrapper.value == 42.0
        assert wrapper.modified
        assert wrapper.valid
        assert not wrapper.truncated
        assert not wrapper.unwritten
    
    def test_repr(self):
        """Test __repr__ output."""
        status = FieldStatus(0b00000100)
        wrapper = ValueWithStatus(3.14, status)
        
        repr_str = repr(wrapper)
        assert "ValueWithStatus" in repr_str
        assert "3.14" in repr_str
        assert "valid=True" in repr_str or "valid" in repr_str
    
    def test_with_different_types(self):
        """Test ValueWithStatus with various value types."""
        status = FieldStatus(0b00000100)
        
        # Float
        wrapper_float = ValueWithStatus(1.5, status)
        assert float(wrapper_float) == 1.5
        
        # Int
        wrapper_int = ValueWithStatus(42, status)
        assert int(wrapper_int) == 42
        
        # String
        wrapper_str = ValueWithStatus("test", status)
        assert str(wrapper_str) == "test"
        
        # Array
        arr = np.array([1, 2, 3])
        wrapper_arr = ValueWithStatus(arr, status)
        np.testing.assert_array_equal(np.array(wrapper_arr), arr)


# Practical usage tests
class TestPracticalUsage:
    """Test practical usage patterns."""
    
    def test_conditional_processing(self, unique_name, cleanup_list):
        """Test typical conditional processing pattern."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=25.5, count=10)
        data = shm.read(timeout=0)
        
        # Typical usage pattern
        temp = data.temperature
        if temp.valid and temp.modified:
            value = temp.value
            assert abs(value - 25.5) < 1e-10
    
    def test_value_and_status_access(self, unique_name, cleanup_list):
        """Test accessing both value and status."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=30.0)
        data = shm.read(timeout=0)
        
        temp = data.temperature
        value = temp.value
        is_valid = temp.valid
        is_modified = temp.modified
        
        assert abs(value - 30.0) < 1e-10
        assert is_valid
        assert is_modified
    
    def test_direct_arithmetic_usage(self, unique_name, cleanup_list):
        """Test using value directly in arithmetic."""
        shm = SharedMemory(ScalarData, name=unique_name, create=True)
        cleanup_list.append(shm)
        
        shm.write(temperature=20.0, count=5)
        data = shm.read(timeout=0)
        
        # Use in calculations
        if data.temperature.valid:
            celsius = data.temperature.value
            fahrenheit = celsius * 9/5 + 32
            assert abs(fahrenheit - 68.0) < 1e-10
        
        if data.count.valid:
            total = data.count.value * 2
            assert total == 10


# Manual test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    