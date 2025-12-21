"""
Test reader auto-detection of configuration from header.

This validates the core feature: readers don't need to know slots count
or any other configuration - everything is in the header!
"""

import pytest
from dataclasses import dataclass
import multiprocessing
from multiprocessing import Queue
import sys

from flexible_shared_memory import SharedMemory


@dataclass
class AutoDetectData:
    value: float = 0.0
    count: int = 0
    message: "str[32]" = ""


PROCESS_START_METHODS = ["fork", "spawn"] if sys.platform != "win32" else ["spawn"]


def reader_auto_detect(name: str, queue: Queue):
    """Reader that auto-detects ALL configuration."""
    try:
        # NO parameters except name - DataClass is auto-reconstructed!
        shm = SharedMemory(name)  # ATTACH
        
        # What did it detect?
        result = {
            "detected_slots": shm.slots,
            "is_fifo": shm.is_fifo,
            "slot_size": shm._slot_size,
            "num_fields": len(shm._layout.fields)
        }
        
        # Try to read
        data = shm.read(timeout=1.0)
        result["data_received"] = data is not None
        if data:
            result["value"] = data.value.value if data.value.valid else None
            result["count"] = data.count.value if data.count.valid else None
            result["message"] = data.message.value if data.message.valid else None
        
        shm.close()
        queue.put(("success", result))
    except Exception as e:
        import traceback
        queue.put(("error", f"{e}\n{traceback.format_exc()}"))


class TestReaderAutoDetect:
    """Test auto-detection of configuration."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    @pytest.mark.parametrize("num_slots", [1, 3, 5, 10])
    def test_reader_detects_slots_automatically(self, start_method, num_slots):
        """Reader auto-detects slots count from header."""
        ctx = multiprocessing.get_context(start_method)
        
        # Writer creates with specific slots
        shm = SharedMemory(AutoDetectData, slots=num_slots)  # CREATE
        name = shm.name  # Get generated name
        
        try:
            # Write some data
            if num_slots == 1:
                shm.write(value=42.0, count=1, message="test")
            else:
                shm.write(value=42.0, count=1, message="test")
                shm.finalize()
            
            # Reader opens WITHOUT specifying anything
            queue = ctx.Queue()
            proc = ctx.Process(target=reader_auto_detect, args=(name, queue))
            proc.start()
            proc.join(timeout=5.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Reader failed: {result}"
            
            # CRITICAL: Reader must detect correct slots!
            assert result["detected_slots"] == num_slots, \
                f"Reader detected {result['detected_slots']} slots, expected {num_slots} (start_method={start_method})"
            
            assert result["is_fifo"] == (num_slots > 1), \
                f"Reader FIFO mode mismatch (start_method={start_method})"
            
            assert result["data_received"], \
                f"Reader failed to receive data (start_method={start_method})"
            
            # Verify data values
            assert result["value"] == 42.0
            assert result["count"] == 1
            assert result["message"] == "test"
        
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_reader_detects_field_layout(self, start_method):
        """Reader auto-detects field layout."""
        ctx = multiprocessing.get_context(start_method)
        
        shm = SharedMemory(AutoDetectData)  # CREATE
        name = shm.name
        
        try:
            shm.write(value=12.34, count=99, message="hello")
            
            queue = ctx.Queue()
            proc = ctx.Process(target=reader_auto_detect, args=(name, queue))
            proc.start()
            proc.join(timeout=5.0)
            
            status, result = queue.get(timeout=1.0)
            assert status == "success", f"Reader failed: {result}"
            
            # Should detect 3 fields
            assert result["num_fields"] == 3
            
            # Should read correct values
            assert abs(result["value"] - 12.34) < 1e-10
            assert result["count"] == 99
            assert result["message"] == "hello"
        
        finally:
            shm.close()
            shm.unlink()
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_multiple_readers_same_config(self, start_method):
        """Multiple readers all auto-detect same configuration."""
        ctx = multiprocessing.get_context(start_method)
        
        # Writer with FIFO
        shm = SharedMemory(AutoDetectData, slots=5)  # CREATE
        name = shm.name
        
        try:
            shm.write(value=1.0, count=1)
            shm.finalize()
            
            # Start 3 readers
            readers = []
            queues = []
            
            for i in range(3):
                q = ctx.Queue()
                p = ctx.Process(target=reader_auto_detect, args=(name, q))
                readers.append(p)
                queues.append(q)
                p.start()
            
            # Wait for all
            for p in readers:
                p.join(timeout=5.0)
            
            # Check all detected same config
            for i, q in enumerate(queues):
                status, result = q.get(timeout=1.0)
                assert status == "success", f"Reader {i} failed: {result}"
                
                assert result["detected_slots"] == 5, \
                    f"Reader {i} detected wrong slots (start_method={start_method})"
                assert result["is_fifo"] is True
        
        finally:
            shm.close()
            shm.unlink()


class TestHeaderValidation:
    """Test header hash validation."""
    
    @pytest.mark.parametrize("start_method", PROCESS_START_METHODS)
    def test_wrong_dataclass_detected(self, start_method):
        """Reader with different dataclass fails hash validation."""
        
        @dataclass
        class WrongDataclass:
            other_field: int = 0
            different: float = 0.0
        
        # Create shared memory with AutoDetectData
        shm = SharedMemory(AutoDetectData)  # CREATE
        name = shm.name
        
        try:
            shm.write(value=1.0)
            
            # Try to validate with wrong dataclass - should fail hash check
            with pytest.raises(ValueError, match="Structure mismatch"):
                SharedMemory(name, expected_type=WrongDataclass)  # ATTACH with validation
        
        finally:
            shm.close()
            shm.unlink()
    
    def test_reader_with_slots_parameter_rejected(self):
        """Reader cannot specify slots parameter (auto-detection only)."""
        
        # Create shared memory
        shm = SharedMemory(AutoDetectData, slots=5)  # CREATE
        name = shm.name
        
        try:
            shm.write(value=1.0)
            
            # Try to attach WITH slots parameter - should fail
            with pytest.raises(ValueError, match="not allowed in ATTACH mode"):
                SharedMemory(name, slots=5)  # ATTACH - cannot specify slots!
            
            # Opening without slots should work
            shm_reader = SharedMemory(name)  # ATTACH - auto-detects
            assert shm_reader.slots == 5  # Auto-detected
            shm_reader.close()
        
        finally:
            shm.close()
            shm.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
