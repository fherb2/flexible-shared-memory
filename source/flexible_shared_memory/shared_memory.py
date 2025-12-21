"""
Lock-Free Shared Memory System with Self-Describing Header
============================================================

This module provides a high-performance, lock-free shared memory system for 
inter-process communication in Python. It automatically maps Python dataclasses 
to shared memory layouts and stores configuration in the header so readers
can auto-detect all settings.

License
-------
MIT License - Copyright (c) 2024 fherb2
https://gitlab.com/fherb2/flexible-shared-memory

Key Feature: Self-Describing Header
------------------------------------
The shared memory contains a header with ALL configuration:
- Number of slots (FIFO vs single-slot)
- Field layout (offsets, types, sizes)
- Dataclass structure

Readers can open with just the name - no configuration needed!

IMPORTANT: Dataclass Compatibility
-----------------------------------
The dataclass structure is FROZEN at shared memory creation time.
All processes (writer and readers) MUST use the EXACT SAME dataclass
definition:
- Same field names
- Same field types  
- Same field order
- Same type annotations (e.g., str[64], float64[10,10])

If the dataclass changes (fields added/removed/reordered), the header hash
will mismatch and readers will raise a ValueError. This is intentional to
prevent silent data corruption.

Header Structure:
    Bytes 0-7:    Hash (lower 8 bytes of SHA256 over header data)
    Bytes 8-11:   Header Length (uint32)
    Bytes 12-19:  Total Length (uint64)
    Bytes 20-23:  slots (uint32)
    Bytes 24+:    Pickled layout info

Example:
    # Writer process
    shm = SharedMemory(SensorData, name="sensors", create=True, slots=5)
    shm.write(temperature=23.5)
    shm.finalize()
    
    # Reader process (auto-detects slots=5!)
    shm = SharedMemory(SensorData, name="sensors")
    data = shm.read()
"""

from dataclasses import dataclass, fields
from multiprocessing import shared_memory
import numpy as np
from typing import Type, Any, Optional
import time
import re
import uuid
import pickle
import struct
import hashlib
import sys


# Header constants
FIXED_HEADER_SIZE = 24  # Hash(8) + HeaderLen(4) + TotalLen(8) + slots(4)


def _check_python_version():
    """Ensure Python version supports stable field ordering."""
    if sys.version_info < (3, 7):
        raise RuntimeError(
            "This module requires Python 3.7+.\n"
            "Earlier versions do not guarantee stable dataclass field order,\n"
            "which is critical for shared memory layout consistency."
        )

_check_python_version()

class FieldStatus:
    """
    Status flags for a single field in shared memory.
    
    Attributes
    ----------
    is_valid : bool
        True if field is an exact copy of source (not truncated or unwritten).
    is_modified : bool
        True if field was written since slot creation or last reset.
    is_truncated : bool
        True if field value was truncated during write.
    is_unwritten : bool
        True if field was never written.
    is_overflow : bool
        True if FIFO overflow caused data loss (FIFO mode only).
    """
    
    MASK_TRUNCATED = 0b00000001
    MASK_UNWRITTEN = 0b00000010
    MASK_MODIFIED = 0b00000100
    MASK_OVERFLOW = 0b00001000
    
    def __init__(self, status_byte: int):
        self._status = status_byte
    
    @property
    def is_truncated(self) -> bool:
        return bool(self._status & self.MASK_TRUNCATED)
    
    @property
    def is_unwritten(self) -> bool:
        return bool(self._status & self.MASK_UNWRITTEN)
    
    @property
    def is_modified(self) -> bool:
        return bool(self._status & self.MASK_MODIFIED)
    
    @property
    def is_overflow(self) -> bool:
        return bool(self._status & self.MASK_OVERFLOW)
    
    @property
    def is_valid(self) -> bool:
        return not (self.is_truncated or self.is_unwritten)

    def _update(self, status_byte: int):
        """Update status from byte (for object pooling)."""
        self._status = status_byte

class ValueWithStatus:
    """Wrapper for field values with status information."""
    
    def __init__(self, value: Any, status: FieldStatus):
        self._value = value
        self._status = status
    
    @property
    def value(self) -> Any:
        return self._value
    
    @property
    def valid(self) -> bool:
        return self._status.is_valid
    
    @property
    def modified(self) -> bool:
        return self._status.is_modified
    
    @property
    def truncated(self) -> bool:
        return self._status.is_truncated
    
    @property
    def unwritten(self) -> bool:
        return self._status.is_unwritten
    
    @property
    def overflow(self) -> bool:
        return self._status.is_overflow
    
    def __float__(self) -> float:
        return float(self._value)
    
    def __int__(self) -> int:
        return int(self._value)
    
    def __bool__(self) -> bool:
        return bool(self._value)
    
    def __str__(self) -> str:
        return str(self._value)
    
    def __repr__(self) -> str:
        return f"ValueWithStatus({self._value!r}, valid={self.valid}, modified={self.modified})"
    
    def __array__(self):
        return np.asarray(self._value)
    
    def __add__(self, other):
        return self._value + other
    
    def __sub__(self, other):
        return self._value - other
    
    def __mul__(self, other):
        return self._value * other
    
    def __truediv__(self, other):
        return self._value / other
    
    def _update(self, value: Any, status: FieldStatus):
        """Update value and status (for object pooling)."""
        self._value = value
        self._status = status


class SharedMemory:
    """
    Lock-free shared memory with self-describing header.
    
    The header contains all configuration, so readers only need the name.
    
    WARNING: Single Writer Only
    ----------------------------
    This implementation is designed for SINGLE-WRITER, MULTIPLE-READER
    scenarios. Using multiple concurrent writers will cause data corruption
    because:
    - No locks are used (lock-free design)
    - Sequence numbers protect against read-during-write, NOT write-during-write
    - Field status flags will be inconsistent
    
    For multi-writer scenarios, use proper process locks (multiprocessing.Lock)
    or a different IPC mechanism.
    
    Parameters
    ----------
    dataclass_type : Type
        Dataclass type defining the data structure.
    name : str, optional
        Shared memory name. If None, generates unique name.
    slots : int, optional
        Number of buffer slots. Only used when create=True.
        Readers auto-detect this from header.
    create : bool, default=False
        If True, creates new shared memory. If False, opens existing.
    
    Examples
    --------
    Writer:
        shm = SharedMemory(SensorData, name="sensors", create=True, slots=5)
        shm.write(temperature=23.5)
        shm.finalize()
    
    Reader (auto-detects slots!):
        shm = SharedMemory(SensorData, name="sensors")
        data = shm.read()
    """
    
    def __init__(self, dataclass_type: Type, name: Optional[str] = None, 
                 slots: Optional[int] = None, create: bool = False):
        
        self.dataclass_type = dataclass_type
        
        # Generate name if not provided
        if name is None:
            name = f"shm_{uuid.uuid4().hex[:8]}"
        
        if create:
            # WRITER: Create with specified configuration
            if slots is None:
                slots = 1
            if slots < 1:
                raise ValueError("slots must be >= 1")
            
            self.slots = slots
            self.is_fifo = slots > 1
            
            # Analyze dataclass structure
            self._layout = _SharedMemoryLayout(dataclass_type)
            self._slot_size = self._layout.total_size
            
            # Build header
            header = self._build_header()
            
            # FIFO metadata size
            metadata_size = 24 if self.is_fifo else 0
            data_size = metadata_size + self._slot_size * slots
            total_size = len(header) + data_size
            
            # Create shared memory
            self.shm = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=total_size
            )
            
            # Write header
            self.shm.buf[0:len(header)] = header
            
            # Set offsets
            self._header_size = len(header)
            self._metadata_offset = self._header_size
            self._data_offset = self._header_size + metadata_size
            
            # Initialize FIFO metadata
            if self.is_fifo:
                self._set_fifo_metadata(0, 0, 0)
            
            # Initialize all slots
            for slot_idx in range(slots):
                self._initialize_slot(slot_idx)
        
        else:
            # READER: Auto-detect configuration from header

            # Reject slots parameter for reader (auto-detection only)
            if slots is not None:
                raise ValueError(
                    "Reader mode does not accept 'slots' parameter.\n"
                    "The number of slots is auto-detected from the shared memory header.\n"
                    "Remove the 'slots' argument when opening existing shared memory."
                )
            
            # Step 1: Read fixed header (24 bytes)
            shm_temp = shared_memory.SharedMemory(name=name)
            
            if shm_temp.size < FIXED_HEADER_SIZE:
                shm_temp.close()
                raise ValueError(f"Shared memory too small: {shm_temp.size} bytes")
            
            stored_hash = int.from_bytes(shm_temp.buf[0:8], 'little')
            header_length = int.from_bytes(shm_temp.buf[8:12], 'little')
            total_length = int.from_bytes(shm_temp.buf[12:20], 'little')
            slots_from_header = int.from_bytes(shm_temp.buf[20:24], 'little')
            
            # Step 2: Read complete header and validate hash
            if shm_temp.size < header_length:
                shm_temp.close()
                raise ValueError(f"Header size mismatch: need {header_length}, have {shm_temp.size}")
            
            header_data = bytes(shm_temp.buf[8:header_length])
            stored_header_hash = self._compute_hash(header_data)
            
            # Step 3: Compute hash from OUR dataclass to verify compatibility
            temp_layout = _SharedMemoryLayout(dataclass_type)
            our_layout_dict = temp_layout.to_dict()
            our_layout_pickle = pickle.dumps(our_layout_dict)
            
            # Build header data exactly as writer would (using actual header_length and total_length)
            our_header_data = bytearray()
            our_header_data.extend(struct.pack('<I', header_length))  # Use actual header length
            our_header_data.extend(struct.pack('<Q', total_length))   # Use actual total length
            our_header_data.extend(struct.pack('<I', slots_from_header))
            our_header_data.extend(our_layout_pickle)
            
            our_computed_hash = self._compute_hash(bytes(our_header_data))
            
            # Compare: our layout hash vs stored layout hash
            if stored_header_hash != our_computed_hash:
                shm_temp.close()
                raise ValueError(
                    f"Dataclass structure mismatch!\n"
                    f"The shared memory was created with a different dataclass definition.\n"
                    f"Ensure all processes use the EXACT SAME dataclass:\n"
                    f"  - Same field names and order\n"
                    f"  - Same types and annotations\n"
                    f"Expected hash: {our_computed_hash:016x}\n"
                    f"Found hash:    {stored_header_hash:016x}"
                )
            
            # Step 4: Parse header (hash validated - safe to use stored layout)
            self.slots = slots_from_header
            self.slots = slots_from_header
            self.is_fifo = self.slots > 1
            
            # Unpickle layout
            layout_pickle = bytes(shm_temp.buf[24:header_length])
            layout_dict = pickle.loads(layout_pickle)
            self._layout = _SharedMemoryLayout.from_dict(layout_dict, dataclass_type)
            self._slot_size = self._layout.total_size
            
            # Verify total size
            metadata_size = 24 if self.is_fifo else 0
            expected_size = header_length + metadata_size + self._slot_size * self.slots
            
            if shm_temp.size < expected_size:
                shm_temp.close()
                raise ValueError(f"Shared memory size mismatch: need {expected_size}, have {shm_temp.size}")
            
            self.shm = shm_temp
            
            # Set offsets
            self._header_size = header_length
            self._metadata_offset = self._header_size
            self._data_offset = self._header_size + metadata_size
        
        self.name = self.shm.name
        
        # Current write buffer for staging (FIFO mode)
        self._write_buffer = {}
        self._write_buffer_dirty = False
        
        # Object pools for performance (avoid allocations in hot path)
        num_fields = len(self._layout.fields)
        self._field_status_pool = [FieldStatus(0) for _ in range(num_fields)]
        self._value_status_pool = [ValueWithStatus(None, FieldStatus(0)) for _ in range(num_fields)]
        self._read_dict = {}  # Reusable dict for reads
    
    def _build_header(self) -> bytes:
        """Build header with hash."""
        # Serialize layout to dict
        layout_dict = self._layout.to_dict()
        layout_pickle = pickle.dumps(layout_dict)
        
        # Build header data (everything after hash)
        header_data = bytearray()
        
        # Header Length (will be filled in later) - 4 bytes
        header_data.extend(b'\x00' * 4)
        
        # Total Length (will be filled in later) - 8 bytes
        header_data.extend(b'\x00' * 8)
        
        # slots - 4 bytes
        header_data.extend(struct.pack('<I', self.slots))
        
        # Pickled layout
        header_data.extend(layout_pickle)
        
        # Now we know header length
        header_length = 8 + len(header_data)  # 8 for hash
        
        # Fill in header length
        struct.pack_into('<I', header_data, 0, header_length)
        
        # Total length will be filled by caller
        # (we don't know data size yet)
        
        # Compute hash over header_data
        hash_value = self._compute_hash(bytes(header_data))
        
        # Build complete header
        complete_header = bytearray()
        complete_header.extend(struct.pack('<Q', hash_value))  # 8 bytes hash
        complete_header.extend(header_data)
        
        return bytes(complete_header)
    
    def _compute_hash(self, data: bytes) -> int:
        """Compute lower 8 bytes of SHA256 as uint64."""
        hash_bytes = hashlib.sha256(data).digest()
        return int.from_bytes(hash_bytes[-8:], 'little')
    
    def write(self, **kwargs):
        """
        Write field values to shared memory.
        
        In single-slot mode, data is written immediately. 
        In FIFO mode, data is staged until finalize().
        """
        if self.is_fifo:
            self._write_buffer.update(kwargs)
            self._write_buffer_dirty = True
        else:
            self._write_to_slot(0, kwargs)
    
    def finalize(self):
        """Finalize staged write in FIFO mode."""
        # Performance: Skip runtime check (design assumption: caller knows mode)
        # if not self.is_fifo:
        #     raise RuntimeError("finalize() only for FIFO (slots > 1)")
        
        if not self._write_buffer_dirty:
            return
        
        write_idx, read_idx, count = self._get_fifo_metadata()
        
        # Check if overflow will occur
        overflow = (count >= self.slots)
        
        slot_idx = write_idx % self.slots
        self._write_to_slot(slot_idx, self._write_buffer, overflow=overflow)
        
        write_idx += 1
        
        if count < self.slots:
            count += 1
        else:
            read_idx += 1
        
        self._set_fifo_metadata(write_idx, read_idx, count)
        
        self._write_buffer.clear()  # Performance: Reuse dict instead of new allocation
        self._write_buffer_dirty = False
    
    def read(self, timeout: float = 0, latest: bool = False, 
            reset_modified: bool = False) -> Optional[Any]:
        """Read data from shared memory.
        
        Parameters
        ----------
        timeout : float, default=0
            Maximum time to wait for data in seconds. 0 = no wait.
        latest : bool, default=False
            In FIFO mode, skip to most recent data.
        reset_modified : bool, default=False
            Clear the 'modified' flag after reading.
            Only supported in single-slot mode (slots=1).
            Raises ValueError in FIFO mode.
        
        Returns
        -------
        Dataclass instance with ValueWithStatus fields, or None if timeout.
        
        Raises
        ------
        ValueError
            If reset_modified=True in FIFO mode (slots > 1).
        
        Notes
        -----
        In FIFO mode, reset_modified is not supported because multiple
        readers could consume the same slot, making the modified flag
        ambiguous (modified for which reader?).
        """
        if reset_modified and self.is_fifo:
            raise ValueError("reset_modified only supported in single-slot mode")
        
        if self.is_fifo:
            return self._read_fifo(timeout, latest)
        else:
            return self._read_single(timeout, reset_modified)
    
    def close(self):
        """Close shared memory connection."""
        self.shm.close()
    
    def unlink(self):
        """Delete shared memory segment from system."""
        self.shm.unlink()
    
    def _initialize_slot(self, slot_idx: int):
        """Initialize slot with unwritten flags."""
        offset = self._get_slot_offset(slot_idx)
        
        self._write_uint64(offset, 0)  # seq_begin
        
        num_fields = len(self._layout.fields)
        status_offset = offset + 8
        for i in range(num_fields):
            self.shm.buf[status_offset + i] = FieldStatus.MASK_UNWRITTEN
        
        seq_end_offset = offset + self._slot_size - 8
        self._write_uint64(seq_end_offset, 0)  # seq_end
    
    def _get_slot_offset(self, slot_idx: int) -> int:
        """Get memory offset for slot."""
        return self._data_offset + slot_idx * self._slot_size
    
    def _read_uint64(self, offset: int) -> int:
        """Read uint64 from shared memory."""
        return int.from_bytes(self.shm.buf[offset:offset+8], 'little')
    
    def _write_uint64(self, offset: int, value: int):
        """Write uint64 to shared memory."""
        self.shm.buf[offset:offset+8] = value.to_bytes(8, 'little')
    
    def _read_uint32(self, offset: int) -> int:
        """Read uint32 from shared memory."""
        return int.from_bytes(self.shm.buf[offset:offset+4], 'little')
    
    def _write_uint32(self, offset: int, value: int):
        """Write uint32 to shared memory."""
        self.shm.buf[offset:offset+4] = value.to_bytes(4, 'little')
    
    def _get_fifo_metadata(self) -> tuple[int, int, int]:
        """Get FIFO write_idx, read_idx, count."""
        if not self.is_fifo:
            return 0, 0, 0
        write_idx = self._read_uint64(self._metadata_offset)
        read_idx = self._read_uint64(self._metadata_offset + 8)
        count = self._read_uint64(self._metadata_offset + 16)
        return write_idx, read_idx, count
    
    def _set_fifo_metadata(self, write_idx: int, read_idx: int, count: int):
        """Set FIFO metadata."""
        if not self.is_fifo:
            return
        self._write_uint64(self._metadata_offset, write_idx)
        self._write_uint64(self._metadata_offset + 8, read_idx)
        self._write_uint64(self._metadata_offset + 16, count)
    
    def _write_to_slot(self, slot_idx: int, data: dict, overflow: bool = False):
        """Write data to slot with sequence numbers."""
        offset = self._get_slot_offset(slot_idx)
        
        seq = self._read_uint64(offset)
        seq += 1
        self._write_uint64(offset, seq)
        
        status_offset = offset + 8
        
        for idx, field_info in enumerate(self._layout.fields):
            field_offset = offset + field_info.offset
            
            if field_info.name in data:
                value = data[field_info.name]
                
                truncated = False
                if field_info.is_scalar:
                    self._write_scalar(field_offset, value, field_info.field_type)
                elif field_info.is_string:
                    truncated = self._write_string(field_offset, value, field_info)
                elif field_info.is_array:
                    truncated = self._write_array(field_offset, value, field_info)
                
                status = self.shm.buf[status_offset + idx]
                status &= ~FieldStatus.MASK_UNWRITTEN
                status |= FieldStatus.MASK_MODIFIED
                
                if truncated:
                    status |= FieldStatus.MASK_TRUNCATED
                else:
                    status &= ~FieldStatus.MASK_TRUNCATED
                
                if overflow:
                    status |= FieldStatus.MASK_OVERFLOW
                else:
                    status &= ~FieldStatus.MASK_OVERFLOW
                
                self.shm.buf[status_offset + idx] = status
            else:
                status = self.shm.buf[status_offset + idx]
                status &= ~FieldStatus.MASK_MODIFIED
                self.shm.buf[status_offset + idx] = status
        
        seq_end_offset = offset + self._slot_size - 8
        self._write_uint64(seq_end_offset, seq)
    
    def _write_scalar(self, offset: int, value: Any, field_type: Type):
        """Write scalar value."""
        if field_type == float:
            np.ndarray(1, dtype=np.float64, buffer=self.shm.buf, offset=offset)[0] = value
        elif field_type == int:
            np.ndarray(1, dtype=np.int32, buffer=self.shm.buf, offset=offset)[0] = value
        elif field_type == bool:
            self.shm.buf[offset] = 1 if value else 0
    
    def _write_string(self, offset: int, value: str, field_info: '_FieldInfo') -> bool:
        """Write string with UTF-8 encoding, return True if truncated."""
        max_bytes = field_info.string_max_bytes  # Performance: Use pre-cached value
        
        # Check character truncation FIRST (before encoding)
        char_truncated = len(value) > field_info.string_max_chars
        if char_truncated:
            value = value[:field_info.string_max_chars]  # Truncate to max characters
        
        encoded = value.encode('utf-8')
        
        # Check byte truncation (should rarely happen with 4x reserve, safety fallback)
        if len(encoded) > max_bytes:
            # Truncate to max_bytes, but ensure valid UTF-8 boundary
            encoded = encoded[:max_bytes]
            # Walk backwards to find valid UTF-8 start
            while len(encoded) > 0:
                try:
                    encoded.decode('utf-8')
                    break
                except UnicodeDecodeError:
                    encoded = encoded[:-1]
            char_truncated = True
        
        self._write_uint32(offset, len(encoded))
        self.shm.buf[offset+4:offset+4+len(encoded)] = encoded
        
        return char_truncated
    
    def _write_array(self, offset: int, value: np.ndarray, field_info: '_FieldInfo') -> bool:
        """Write array, return True if truncated."""
        value = np.asarray(value, dtype=field_info.array_dtype)
        truncated = value.shape != field_info.array_shape
        
        flat_value = value.flatten()
        expected_size = field_info.array_flat_size  # Performance: Use pre-cached value
        
        if len(flat_value) > expected_size:
            flat_value = flat_value[:expected_size]
            truncated = True
        elif len(flat_value) < expected_size:
            flat_value = np.pad(flat_value, (0, expected_size - len(flat_value)))
        
        target = np.ndarray(
            expected_size,
            dtype=field_info.array_dtype,
            buffer=self.shm.buf,
            offset=offset
        )
        target[:] = flat_value
        
        return truncated
    
    def _read_single(self, timeout: float, reset_modified: bool) -> Optional[Any]:
        """Read from single slot."""
        start_time = time.time()
        
        while True:
            data = self._read_from_slot(0, reset_modified)
            if data is not None:
                return data
            
            if timeout == 0 or (time.time() - start_time) >= timeout:
                return None
            
            time.sleep(0.0001)
    
    def _read_fifo(self, timeout: float, latest: bool) -> Optional[Any]:
        """Read from FIFO."""
        start_time = time.time()
        
        while True:
            write_idx, read_idx, count = self._get_fifo_metadata()
            
            if count == 0:
                if timeout == 0 or (time.time() - start_time) >= timeout:
                    return None
                time.sleep(0.0001)
                continue
            
            if latest and count > 1:
                read_idx = write_idx - 1
                count = 1
            
            slot_idx = read_idx % self.slots
            data = self._read_from_slot(slot_idx, False)
            
            if data is None:
                time.sleep(0.0001)
                continue
            
            read_idx += 1
            count -= 1
            self._set_fifo_metadata(write_idx, read_idx, count)
            
            return data
    
    def _read_from_slot(self, slot_idx: int, reset_modified: bool) -> Optional[Any]:
        """Read from slot with sequence check."""
        offset = self._get_slot_offset(slot_idx)
        
        seq_begin = self._read_uint64(offset)
        status_offset = offset + 8
        
        # Performance: Reuse dict and pooled objects instead of allocating new ones
        self._read_dict.clear()
        for idx, field_info in enumerate(self._layout.fields):
            field_offset = offset + field_info.offset
            
            status_byte = self.shm.buf[status_offset + idx]
            status_obj = self._field_status_pool[idx]
            status_obj._update(status_byte)
            
            if field_info.is_scalar:
                value = self._read_scalar(field_offset, field_info.field_type)
            elif field_info.is_string:
                value = self._read_string(field_offset, field_info)
            elif field_info.is_array:
                value = self._read_array(field_offset, field_info)
            
            wrapper = self._value_status_pool[idx]
            wrapper._update(value, status_obj)
            self._read_dict[field_info.name] = wrapper
        
        seq_end_offset = offset + self._slot_size - 8
        seq_end = self._read_uint64(seq_end_offset)
        
        if seq_begin != seq_end:
            return None
        
        if reset_modified:
            for idx in range(len(self._layout.fields)):
                status_byte = self.shm.buf[status_offset + idx]
                status_byte &= ~FieldStatus.MASK_MODIFIED
                self.shm.buf[status_offset + idx] = status_byte
        
        # Check if at least one field is valid (not unwritten)
        # If all fields are unwritten, return None (no data available yet)
        has_valid_data = any(not wrapper._status.is_unwritten for wrapper in self._read_dict.values())
        if not has_valid_data:
            return None
        
        return self.dataclass_type(**self._read_dict)
    
    def _read_scalar(self, offset: int, field_type: Type) -> Any:
        """Read scalar value."""
        if field_type == float:
            return float(np.ndarray(1, dtype=np.float64, buffer=self.shm.buf, offset=offset)[0])
        elif field_type == int:
            return int(np.ndarray(1, dtype=np.int32, buffer=self.shm.buf, offset=offset)[0])
        elif field_type == bool:
            return bool(self.shm.buf[offset])
    
    def _read_string(self, offset: int, field_info: '_FieldInfo') -> str:
        """Read UTF-8 string."""
        length = self._read_uint32(offset)
        encoded = bytes(self.shm.buf[offset+4:offset+4+length])
        return encoded.decode('utf-8', errors='ignore')
    
    def _read_array(self, offset: int, field_info: '_FieldInfo') -> np.ndarray:
        """Read array."""
        size = field_info.array_flat_size  # Performance: Use pre-cached value
        flat_array = np.ndarray(
            size,
            dtype=field_info.array_dtype,
            buffer=self.shm.buf,
            offset=offset
        ).copy()
        
        return flat_array.reshape(field_info.array_shape)


class _SharedMemoryLayout:
    """Calculate memory layout for dataclass."""
    
    def __init__(self, dataclass_type: Type):
        self.dataclass_type = dataclass_type
        self.fields: list['_FieldInfo'] = []
        self.total_size = 0
        
        self._analyze_fields()
        self._calculate_layout()
    
    def _analyze_fields(self):
        """Extract and analyze all fields."""
        for field in fields(self.dataclass_type):
            field_info = _FieldInfo(
                name=field.name,
                field_type=field.type,
                default=field.default if hasattr(field, 'default') else None
            )
            self.fields.append(field_info)
    
    def _calculate_layout(self):
        """Calculate offsets and total size."""
        num_fields = len(self.fields)
        offset = 8 + num_fields
        offset = (offset + 7) // 8 * 8
        
        for field_info in self.fields:
            field_info.offset = offset
            offset += field_info.size
        
        offset += 8
        self.total_size = (offset + 7) // 8 * 8
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for pickling."""
        return {
            'fields': [f.to_dict() for f in self.fields],
            'total_size': self.total_size
        }
    
    @classmethod
    def from_dict(cls, data: dict, dataclass_type: Type) -> '_SharedMemoryLayout':
        """Deserialize from dictionary."""
        layout = cls.__new__(cls)
        layout.dataclass_type = dataclass_type
        layout.fields = [_FieldInfo.from_dict(f) for f in data['fields']]
        layout.total_size = data['total_size']
        return layout


class _FieldInfo:
    """
    Information about a dataclass field.
    
    Notes
    -----
    Arrays are treated as atomic units:
    - The entire array shares ONE status byte
    - Modifying any element marks the whole array as modified
    - Padding with zeros fills to expected shape but is still valid
    
    For per-element status tracking, use multiple separate fields or
    multiple SharedMemory instances.
    """
    
    def __init__(self, name: str, field_type: Any, default: Any):
        self.name = name
        self.field_type = field_type
        self.default = default
        self.is_scalar = False
        self.is_string = False
        self.is_array = False
        self.string_max_chars = 0
        self.string_max_bytes = 0  # Vorcachen
        self.array_dtype = None
        self.array_shape = None
        self.array_flat_size = 0  # Vorcachen
        self.size = 0
        self.offset = 0
        
        self._parse_type()
    
    def _parse_type(self):
        """Determine field properties."""
        annotation = str(self.field_type)
        
        string_chars = _AnnotationParser.parse_string(annotation)
        if string_chars:
            self.is_string = True
            self.string_max_chars = string_chars
            self.string_max_bytes = string_chars * 4  # Vorcachen für Performance
            self.size = 4 + self.string_max_bytes
            return
        
        array_info = _AnnotationParser.parse_array(annotation)
        if array_info:
            self.is_array = True
            self.array_dtype, self.array_shape = array_info
            self.array_flat_size = int(np.prod(self.array_shape))  # Vorcachen für Performance
            self.size = self.array_flat_size * np.dtype(self.array_dtype).itemsize
            return
        
        type_sizes = {float: 8, int: 4, bool: 1}
        
        if self.field_type in type_sizes:
            self.is_scalar = True
            self.size = type_sizes[self.field_type]
            return
        
        raise ValueError(f"Unsupported type for field '{self.name}': {self.field_type}")
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'field_type': self.field_type.__name__ if hasattr(self.field_type, '__name__') else str(self.field_type),
            'is_scalar': self.is_scalar,
            'is_string': self.is_string,
            'is_array': self.is_array,
            'string_max_chars': self.string_max_chars,
            'string_max_bytes': self.string_max_bytes,  # Neu
            'array_dtype': np.dtype(self.array_dtype).name if self.array_dtype else None,
            'array_shape': self.array_shape,
            'array_flat_size': self.array_flat_size,  # Neu
            'size': self.size,
            'offset': self.offset
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> '_FieldInfo':
        """Deserialize from dictionary."""
        field_info = cls.__new__(cls)
        field_info.name = data['name']
        
        # Reconstruct field_type safely
        type_name = data['field_type']
        type_map = {'float': float, 'int': int, 'bool': bool}
        field_info.field_type = type_map.get(type_name, str)  # ← GEÄNDERT: Fallback zu str statt eval
        
        field_info.is_scalar = data['is_scalar']
        field_info.is_string = data['is_string']
        field_info.is_array = data['is_array']
        field_info.string_max_chars = data['string_max_chars']
        field_info.string_max_bytes = data.get('string_max_bytes', data['string_max_chars'] * 4)  # Backwards compatibility
        
        if data['array_dtype']:
            # Parse numpy dtype string safely
            dtype_str = data['array_dtype'].replace('dtype(', '').replace(')', '').strip("'\"")
            field_info.array_dtype = np.dtype(dtype_str)  # ← numpy.dtype() ist sicher!
        else:
            field_info.array_dtype = None
        
        field_info.array_shape = data['array_shape']
        field_info.array_flat_size = data.get('array_flat_size', 0)  # Backwards compatibility
        if field_info.array_flat_size == 0 and field_info.array_shape:
            field_info.array_flat_size = int(np.prod(field_info.array_shape))
        field_info.size = data['size']
        field_info.offset = data['offset']
        field_info.default = None
        
        return field_info


class _AnnotationParser:
    """Parse type annotations."""
    
    @staticmethod
    def parse_string(annotation: str) -> Optional[int]:
        """Parse 'str[64]' -> 64 (character count)."""
        match = re.match(r'str\[(\d+)\]', annotation)
        return int(match.group(1)) if match else None
    
    @staticmethod
    def parse_array(annotation: str) -> Optional[tuple]:
        """Parse 'float64[480,640,3]' -> (np.float64, (480,640,3))."""
        match = re.match(r'(float\d+|int\d+|uint\d+|bool)\[([\d,]+)\]', annotation)
        if not match:
            return None
        
        dtype_str, shape_str = match.groups()
        
        dtype_map = {
            'float32': np.float32, 'float64': np.float64,
            'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64,
            'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
            'bool': np.bool_,
        }
        dtype = dtype_map.get(dtype_str, np.float64)
        shape = tuple(int(x) for x in shape_str.split(','))
        
        return dtype, shape
