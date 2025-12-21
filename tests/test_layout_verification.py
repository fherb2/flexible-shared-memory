"""
Layout verification and header integrity tests.

Tests that the self-describing header works correctly:
- Hash validation prevents corruption
- Layout is consistent across processes
- Field order is preserved
"""

import pytest
from dataclasses import dataclass
import multiprocessing
from multiprocessing import Queue
import sys

from flexible_shared_memory import SharedMemory


@dataclass
class LayoutTestData:
    temperature: float = 0.0
    pressure: float = 0.0
    count: int = 0
    active: bool = False
    message: "str[32]" = ""


PROCESS_START_METHODS = ["fork", "spawn"] if sys.platform != "win32" else ["spawn"]


def extract_layout_info(name: str, queue: Queue):
    """Extract layout info from subprocess."""
    try:
        shm = SharedMemory(name)  # ATTACH - auto-detects everything
        
        layout_info = {
            'total_size': shm._layout.total_size,
            'slot_size': shm._slot_size,
            'num_fields': len(shm._layout.fields),
            'slots': shm.slots,
            'is_fifo': shm.is_fifo,
            'fields': []
        }
        
        for field in shm._layout.fields:
            layout_info['fields'].append({
                'name': field.name,
                'offset': field.offset,
                'size': field.size,
            })
        
        shm.close()
        queue.put(("success", layout_info))
    except Exception as e:
        import traceback
        queue.put(("error", f"{e}\n{traceback.format_exc()}"))


def write_known_pattern(name: str, queue: Queue):
    """Write known pattern."""
    try:
        shm = SharedMemory(name)  # ATTACH
        
        shm.write(
            temperature=12.34,
            pressure=56.78,
            count=42,
            active=True,
            message="TestPattern"
        )
        
        shm.close()
        queue.put(("success", None))
    except Exception as e:
        queue.put(("error", str(e)))


class TestLayoutConsistency:
    """Test layout consistency across processes."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    @pytest.mark.parametrize("num_slots", [1, 5])
    def test_layout_matches_across_processes(self, start_method, num_slots):
        """Layout is identical in parent and child."""
        ctx = multiprocessing.get_context(start_method)
        
        shm = SharedMemory(LayoutTestData, slots=num_slots)  # CREATE
        name = shm.name  # Get generated name
        
        try:
            # Parent layout
            parent_layout = {
                'total_size': shm._layout.total_size,
                'slot_size': shm._slot_size,
                'num_fields': len(shm._layout.fields),
                'slots': shm.slots,
                'fields': [(f.name, f.offset, f.size) for f in shm._layout.fields]
            }
            
            # Child layout
            queue = ctx.Queue()
            proc = ctx.Process(target=extract_layout_info, args=(name, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, child_layout = queue.get(timeout=1.0)
            assert status == "success", f"Child failed: {child_layout}"
            
            # Compare
            assert parent_layout['total_size'] == child_layout['total_size']
            assert parent_layout['slot_size'] == child_layout['slot_size']
            assert parent_layout['slots'] == child_layout['slots']
            
            # Check fields
            for i, (pf, cf) in enumerate(zip(parent_layout['fields'], child_layout['fields'])):
                p_name, p_offset, p_size = pf
                c_name = cf['name']
                c_offset = cf['offset']
                c_size = cf['size']
                
                assert p_name == c_name, \
                    f"Field {i} name mismatch: {p_name} != {c_name}"
                assert p_offset == c_offset, \
                    f"Field {p_name} offset mismatch: {p_offset} != {c_offset}"
                assert p_size == c_size, \
                    f"Field {p_name} size mismatch: {p_size} != {c_size}"
        
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_data_transmission_with_known_pattern(self, start_method):
        """Known data pattern is correctly transmitted."""
        ctx = multiprocessing.get_context(start_method)
        
        shm = SharedMemory(LayoutTestData)  # CREATE
        name = shm.name
        
        try:
            # Write from child
            queue = ctx.Queue()
            proc = ctx.Process(target=write_known_pattern, args=(name, queue))
            proc.start()
            proc.join(timeout=2.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Writer failed: {result}"
            
            # Read in parent
            data = shm.read(timeout=1.0)
            assert data is not None
            
            assert data.temperature.valid
            assert abs(data.temperature.value - 12.34) < 1e-10
            
            assert data.pressure.valid
            assert abs(data.pressure.value - 56.78) < 1e-10
            
            assert data.count.valid
            assert data.count.value == 42
            
            assert data.active.valid
            assert data.active.value is True
            
            assert data.message.valid
            assert data.message.value == "TestPattern"
        
        finally:
            shm.close()
            shm.unlink()


class TestHeaderIntegrity:
    """Test header hash validation."""
    
    def test_header_hash_protects_against_wrong_dataclass(self):
        """Hash validation detects wrong dataclass structure."""
        
        @dataclass
        class DifferentData:
            value: float = 0.0
            other: int = 0
        
        # Create with LayoutTestData
        shm = SharedMemory(LayoutTestData)  # CREATE
        name = shm.name
        shm.write(temperature=25.0)
        
        try:
            # Try to open with different dataclass - should fail hash check
            with pytest.raises(ValueError, match="Structure mismatch"):
                SharedMemory(name, expected_type=DifferentData)
        
        finally:
            shm.close()
            shm.unlink()
    
    def test_valid_header_passes_check(self):
        """Valid header passes hash check."""
        
        shm1 = SharedMemory(LayoutTestData)  # CREATE
        name = shm1.name
        shm1.write(temperature=1.0)
        
        try:
            # Should open fine with expected_type validation
            shm2 = SharedMemory(name, expected_type=LayoutTestData)
            data = shm2.read(timeout=0)
            
            assert data is not None
            assert abs(data.temperature.value - 1.0) < 1e-10
            
            shm2.close()
        
        finally:
            shm1.close()
            shm1.unlink()

    def test_field_order_is_deterministic(self):
        """Field order is deterministic across multiple instantiations."""
        
        # Create multiple instances and check field order
        shm1 = SharedMemory(LayoutTestData)  # CREATE
        name = shm1.name
        
        try:
            # Extract field order from first instance
            fields1 = [(f.name, f.offset, f.size) for f in shm1._layout.fields]
            
            # Close and create new instance (ATTACH)
            shm1.close()
            
            shm2 = SharedMemory(name)  # ATTACH
            fields2 = [(f.name, f.offset, f.size) for f in shm2._layout.fields]
            shm2.close()
            
            # Field order must be identical
            assert fields1 == fields2, \
                "Field order must be deterministic across instantiations"
            
            # Verify expected field names in expected order
            expected_names = ['temperature', 'pressure', 'count', 'active', 'message']
            actual_names = [f[0] for f in fields1]
            
            assert actual_names == expected_names, \
                f"Expected field order {expected_names}, got {actual_names}"
        
        finally:
            shm1.unlink()
    
    def test_different_dataclass_different_hash(self):
        """Different dataclasses produce different header hashes."""
        
        @dataclass
        class DataA:
            field1: float = 0.0
            field2: int = 0
        
        @dataclass
        class DataB:
            field1: float = 0.0
            field3: int = 0  # Different field name
        
        shm_a = SharedMemory(DataA)  # CREATE
        shm_b = SharedMemory(DataB)  # CREATE
        
        try:
            # Extract hashes from headers
            hash_a = int.from_bytes(shm_a.shm.buf[0:8], 'little')
            hash_b = int.from_bytes(shm_b.shm.buf[0:8], 'little')
            
            # Hashes must be different for different structures
            assert hash_a != hash_b, \
                "Different dataclass structures must produce different hashes"
        
        finally:
            shm_a.close()
            shm_a.unlink()
            shm_b.close()
            shm_b.unlink()
    
    def test_same_dataclass_same_hash(self):
        """Same dataclass produces same hash."""
        
        @dataclass
        class ConsistentData:
            x: float = 0.0
            y: int = 0
        
        shm1 = SharedMemory(ConsistentData)  # CREATE
        shm2 = SharedMemory(ConsistentData)  # CREATE (different instance)
        
        try:
            # Extract hashes
            hash1 = int.from_bytes(shm1.shm.buf[0:8], 'little')
            hash2 = int.from_bytes(shm2.shm.buf[0:8], 'little')
            
            # Hashes must be identical for same structure
            assert hash1 == hash2, \
                "Same dataclass structure must produce identical hashes"
        
        finally:
            shm1.close()
            shm1.unlink()
            shm2.close()
            shm2.unlink()


class TestLayoutEdgeCases:
    """Test edge cases in layout handling."""
    
    def test_unsupported_field_type_raises_error(self):
        """Test that unsupported field types raise ValueError."""
        
        @dataclass
        class UnsupportedData:
            value: dict = None  # dict is not supported!
        
        # Should raise ValueError during layout analysis
        with pytest.raises(ValueError, match="Unsupported type"):
            SharedMemory(UnsupportedData)  # CREATE



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
