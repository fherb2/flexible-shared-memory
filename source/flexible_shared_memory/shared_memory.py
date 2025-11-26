"""
Lock-Free Shared Memory System with Automatic Type Mapping
===========================================================

This module provides a high-performance, lock-free shared memory system for 
inter-process communication in Python. It automatically maps Python dataclasses 
to shared memory layouts and supports both single-slot and FIFO (ring buffer) modes.

License
-------
MIT License - Copyright (c) 2024 fherb2
https://gitlab.com/fherb2/flexible-shared-memory

Features
--------
- **Automatic type mapping**: Define data structure as dataclass, shared memory 
  layout is generated automatically
- **Lock-free operation**: Uses sequence numbers for consistent reads without locks
- **NumPy integration**: Supports multi-dimensional arrays for images, oscilloscope 
  data, etc.
- **String support**: UTF-8 encoded strings with character-count limits
- **Field-level status**: Each field tracks valid, modified, truncated, unwritten state
- **FIFO mode**: Ring buffer with configurable slots for buffered communication
- **Single-slot mode**: Minimal latency for single value updates

Type Annotations
----------------
Define field types using standard Python types and custom annotations:

Scalar types:
    - `float` → 64-bit float (numpy.float64)
    - `int` → 32-bit integer (numpy.int32)  
    - `bool` → boolean

Strings (UTF-8, character-count based):
    - `"str[64]"` → Max 64 Unicode characters (allocates 260 bytes)
    - Supports all Unicode: ASCII, Cyrillic, Chinese, Japanese, Emojis

Arrays:
    - `"float32[480,640,3]"` → 3D array of shape (480, 640, 3)
    - Supported dtypes: float32, float64, int8, int16, int32, int64, 
      uint8, uint16, uint32, uint64, bool

Examples
--------
Basic single-slot usage:

>>> from dataclasses import dataclass
>>> import numpy as np
>>> 
>>> @dataclass
>>> class SensorData:
...     temperature: float = 0.0
...     pressure: float = 0.0
...     timestamp: float = 0.0
...     status_msg: "str[32]" = ""
>>> 
>>> # Writer process
>>> shm = SharedMemory(SensorData, name="sensors")
>>> shm.write(temperature=23.5, pressure=1013.25, status_msg="OK")
>>> data = shm.read(timeout=0)
>>> 
>>> # Access values with status
>>> temp = data.temperature
>>> if temp.valid and temp.modified:
...     print(f"Temperature: {temp.value}°C")
>>> 
>>> # Handle invalid data
>>> if not temp.valid:
...     if temp.truncated:
...         print("Error: Temperature data was truncated!")
...     elif temp.unwritten:
...         print("Warning: Temperature never written")
>>> 
>>> shm.close()
>>> shm.unlink()

Field status checking:

>>> data = shm.read()
>>> pos = data.position
>>> 
>>> # Check individual status flags
>>> if pos.valid:
...     use(pos.value)
>>> 
>>> if pos.modified:
...     print("Position changed!")
>>> 
>>> if pos.truncated:
...     print("Warning: position was truncated")
>>> 
>>> if pos.unwritten:
...     print("Position never written")
>>> 
>>> # Auto-conversion with magic methods
>>> x = float(pos)  # Automatic conversion
>>> y = pos.value   # Explicit access

FIFO usage with buffering:

>>> @dataclass
>>> class ControllerState:
...     position: float = 0.0
...     velocity: float = 0.0
...     command: "str[16]" = ""
>>> 
>>> # Writer process - fast control loop
>>> fifo = SharedMemory(ControllerState, name="ctrl", slots=10)
>>> for i in range(100):
...     fifo.write(position=i*0.1, velocity=i*0.05)
...     fifo.finalize()  # Make data available atomically
>>> fifo.close()
>>> fifo.unlink()
>>> 
>>> # Reader process - monitoring at lower rate
>>> fifo = SharedMemory(ControllerState, name="ctrl", slots=10, create=False)
>>> data = fifo.read(timeout=1.0, latest=False)
>>> if data.position.modified:
...     print(f"New position: {data.position.value}")
>>> fifo.close()

Reset modified flag for single-reader scenarios:

>>> # Single reader that wants to track changes
>>> shm = SharedMemory(SensorData, name="sensor1", create=False)
>>> while True:
...     data = shm.read(timeout=1.0, reset_modified=True)
...     if data.temperature.modified:
...         print("Temperature changed!")

Image/array transfer:

>>> @dataclass
>>> class ImageFrame:
...     frame_id: int = 0
...     timestamp: float = 0.0
...     image: "uint8[480,640,3]" = None
>>> 
>>> shm = SharedMemory(ImageFrame, name="camera")
>>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
>>> shm.write(frame_id=42, timestamp=time.time(), image=frame)
>>> 
>>> data = shm.read(timeout=0.5)
>>> if data.image.valid and not data.image.truncated:
...     img = data.image.value  # NumPy array
...     print(img.shape)

Notes
-----
- Field status flags persist until next write (consistent for all readers)
- FIFO overwrites oldest data when full (no blocking)
- Reading blocks until data available or timeout expires
- Sequence numbers ensure consistent reads without locks
- String and array truncation sets truncated flag per field
"""

from dataclasses import dataclass, fields
from multiprocessing import shared_memory
import numpy as np
from typing import Type, Any, Optional
import time
import re
import uuid


class FieldStatus:
    """
    Status flags for a single field in shared memory.
    
    Provides information about validity, modifications, and truncation
    for individual fields.
    
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
    
    Notes
    -----
    A field is only valid if it contains the complete, exact data from the source.
    Truncated data may be unusable (e.g., incomplete polynomial coefficients).
    Users must check truncated/unwritten flags to decide how to handle invalid data.
    
    Examples
    --------
    >>> status = data.position_status
    >>> if status.is_valid and status.is_modified:
    ...     process_update(data.position)
    >>> elif status.is_truncated:
    ...     log.error("Position data truncated - cannot use!")
    >>> elif status.is_unwritten:
    ...     log.warning("No position data available")
    """
    
    # Bit masks for status byte
    MASK_TRUNCATED = 0b00000001
    MASK_UNWRITTEN = 0b00000010
    MASK_MODIFIED = 0b00000100
    
    def __init__(self, status_byte: int):
        self._status = status_byte
    
    @property
    def is_truncated(self) -> bool:
        """True if field was truncated during write."""
        return bool(self._status & self.MASK_TRUNCATED)
    
    @property
    def is_unwritten(self) -> bool:
        """True if field was never written."""
        return bool(self._status & self.MASK_UNWRITTEN)
    
    @property
    def is_modified(self) -> bool:
        """True if field was written since slot/reset."""
        return bool(self._status & self.MASK_MODIFIED)
    
    @property
    def is_valid(self) -> bool:
        """True if field is an exact copy of source (not truncated or unwritten)."""
        return not (self.is_truncated or self.is_unwritten)


class ValueWithStatus:
    """
    Wrapper for field values with status information.
    
    Provides pythonic access to both the field value and its status flags.
    
    Parameters
    ----------
    value : any
        The actual field value (float, int, str, np.ndarray, etc.)
    status : FieldStatus
        Status information for this field
    
    Attributes
    ----------
    value : any
        The actual field value
    valid : bool
        True if field is an exact copy of source (not truncated or unwritten)
    modified : bool
        Shortcut for status.is_modified
    truncated : bool
        Shortcut for status.is_truncated
    unwritten : bool
        Shortcut for status.is_unwritten
    
    Notes
    -----
    Always check 'valid' first. If False, check 'truncated' and 'unwritten' to
    determine the appropriate error handling strategy.
    
    Examples
    --------
    >>> pos = data.position
    >>> if pos.valid and pos.modified:
    ...     x = pos.value  # Exact copy from source
    ...     y = float(pos)  # Magic conversion
    >>> elif pos.truncated:
    ...     # Handle error - data may be unusable
    ...     log.error("Position data incomplete!")
    >>> 
    >>> # Works with all types
    >>> msg = data.message
    >>> if msg.valid:
    ...     text = msg.value  # str
    >>> 
    >>> img = data.image
    >>> if img.valid:
    ...     arr = img.value  # np.ndarray
    ...     arr2 = np.array(img)  # Magic
    """
    
    def __init__(self, value: Any, status: FieldStatus):
        self._value = value
        self._status = status
    
    @property
    def value(self) -> Any:
        """The actual field value."""
        return self._value
    
    @property
    def valid(self) -> bool:
        """True if field is an exact copy of source (not truncated or unwritten)."""
        return self._status.is_valid
    
    @property
    def modified(self) -> bool:
        """True if written since slot creation or last reset."""
        return self._status.is_modified
    
    @property
    def truncated(self) -> bool:
        """True if value was truncated during write."""
        return self._status.is_truncated
    
    @property
    def unwritten(self) -> bool:
        """True if never written."""
        return self._status.is_unwritten
    
    # Magic methods for automatic conversion
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
    
    # NumPy support
    def __array__(self):
        return np.asarray(self._value)
    
    # Arithmetic operations
    def __add__(self, other):
        return self._value + other
    
    def __sub__(self, other):
        return self._value - other
    
    def __mul__(self, other):
        return self._value * other
    
    def __truediv__(self, other):
        return self._value / other


class SharedMemory:
    """
    Lock-free shared memory for inter-process communication.
    
    Supports single-slot (minimal latency) and multi-slot FIFO (buffered) modes.
    Automatically maps Python dataclasses to shared memory layouts with sequence
    numbers for consistent lock-free reading. Each field has individual status
    tracking (valid, modified, truncated, unwritten).
    
    Parameters
    ----------
    dataclass_type : Type
        Dataclass type defining the data structure. Should NOT include any
        status fields - these are handled automatically.
    name : str, optional
        Shared memory name. If None, generates unique name accessible via `.name`.
    slots : int, default=1
        Number of buffer slots. 1 for single-slot mode, >1 for FIFO mode.
    create : bool, default=True
        If True, creates new shared memory. If False, opens existing.
    
    Attributes
    ----------
    name : str
        Shared memory identifier, can be passed to other processes.
    slots : int
        Number of buffer slots.
    is_fifo : bool
        True if slots > 1 (FIFO mode), False otherwise.
    
    Examples
    --------
    Single-slot mode (direct write):
    
    >>> shm = SharedMemory(MyDataClass, name="sensor1")
    >>> shm.write(temperature=23.5, pressure=1013.0)
    >>> data = shm.read(timeout=0)
    >>> if data.temperature.valid:
    ...     print(data.temperature.value)
    
    FIFO mode (staged write):
    
    >>> fifo = SharedMemory(MyDataClass, name="buffer", slots=10)
    >>> fifo.write(value1=1.0)
    >>> fifo.write(value2=2.0)
    >>> fifo.finalize()  # Commit atomically
    >>> data = fifo.read(timeout=1.0, latest=False)
    
    Notes
    -----
    - Single-slot: write() commits immediately, no finalize() needed
    - FIFO: write() stages data, finalize() commits atomically
    - Always call close() before process exits
    - Only creator should call unlink()
    - Field status persists until next write (all readers see same state)
    """
    
    def __init__(self, dataclass_type: Type, name: Optional[str] = None, 
                 slots: int = 1, create: bool = True):
        if slots < 1:
            raise ValueError("slots must be >= 1")
        
        self.dataclass_type = dataclass_type
        self.slots = slots
        self.is_fifo = slots > 1
        
        # Generate name if not provided
        if name is None:
            name = f"shm_{uuid.uuid4().hex[:8]}"
        
        # Analyze dataclass structure
        self._layout = _SharedMemoryLayout(dataclass_type)
        self._slot_size = self._layout.total_size
        
        # FIFO metadata size: write_idx(8) + read_idx(8) + count(8) = 24 bytes
        metadata_size = 24 if self.is_fifo else 0
        total_size = metadata_size + self._slot_size * slots
        
        # Set offsets BEFORE using them
        self._metadata_offset = 0
        self._data_offset = metadata_size
        
        # Create or open shared memory
        if create:
            self.shm = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=total_size
            )
            # Initialize FIFO metadata
            if self.is_fifo:
                self._set_fifo_metadata(0, 0, 0)
            
            # Initialize all slots with unwritten flags
            for slot_idx in range(slots):
                self._initialize_slot(slot_idx)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        
        self.name = self.shm.name
        
        # Current write buffer for staging (FIFO mode)
        self._write_buffer = {}
        self._write_buffer_dirty = False
    
    def write(self, **kwargs):
        """
        Write field values to shared memory.
        
        In single-slot mode, data is written immediately. In FIFO mode, data is
        staged in a buffer until finalize() is called. Sets modified flag for
        written fields.
        
        Parameters
        ----------
        **kwargs
            Field names and values to write. Field names must match dataclass fields.
            Values are automatically converted and truncated if necessary.
        
        Examples
        --------
        >>> shm.write(temperature=23.5, pressure=1013.25, status="OK")
        >>> shm.write(image=np.random.rand(480, 640, 3))
        
        Notes
        -----
        - Strings longer than field size are truncated, sets truncated flag
        - Arrays larger than field size are truncated, sets truncated flag
        - In FIFO mode, must call finalize() to commit
        - In single-slot mode, writes are immediately visible
        - Sets modified flag for all written fields
        """
        if self.is_fifo:
            # Stage data in buffer
            self._write_buffer.update(kwargs)
            self._write_buffer_dirty = True
        else:
            # Write directly to slot 0
            self._write_to_slot(0, kwargs)
    
    def finalize(self):
        """
        Finalize staged write in FIFO mode.
        
        Makes buffered data atomically available for reading. Only used in FIFO
        mode (slots > 1). In single-slot mode, this raises an error.
        
        Raises
        ------
        RuntimeError
            If called in single-slot mode (slots=1).
        
        Examples
        --------
        >>> fifo.write(position=1.5, velocity=2.0)
        >>> fifo.write(timestamp=time.time())
        >>> fifo.finalize()  # Now all values are atomically visible
        
        Notes
        -----
        - Advances write pointer and makes data available
        - If buffer is full, overwrites oldest unread data
        - Clears write buffer after commit
        - Sets modified flag for written fields
        """
        if not self.is_fifo:
            raise RuntimeError("finalize() only for FIFO (slots > 1)")
        
        if not self._write_buffer_dirty:
            return
        
        # Get current FIFO state
        write_idx, read_idx, count = self._get_fifo_metadata()
        
        # Write to current write slot
        slot_idx = write_idx % self.slots
        self._write_to_slot(slot_idx, self._write_buffer)
        
        # Advance write pointer
        write_idx += 1
        
        # Update count and potentially advance read pointer
        if count < self.slots:
            count += 1
        else:
            # Buffer full, overwrite oldest
            read_idx += 1
        
        self._set_fifo_metadata(write_idx, read_idx, count)
        
        # Clear buffer
        self._write_buffer = {}
        self._write_buffer_dirty = False
    
    def read(self, timeout: float = 0, latest: bool = False, 
             reset_modified: bool = False) -> Optional[Any]:
        """
        Read data from shared memory.
        
        Blocks until data is available or timeout expires. Uses sequence numbers
        to ensure consistent reads without locks. Returns dataclass instance with
        all fields wrapped in ValueWithStatus.
        
        Parameters
        ----------
        timeout : float, default=0
            Maximum wait time in seconds. 0 means non-blocking (return None if empty).
        latest : bool, default=False
            If True (FIFO only), skips to most recent data, discarding older entries.
            Useful for monitoring at lower rates than write rate.
        reset_modified : bool, default=False
            If True (single-slot only), resets modified flags after reading.
            Useful for single-reader scenarios to track changes.
        
        Returns
        -------
        dataclass instance or None
            Instance with all fields wrapped as ValueWithStatus objects, or None
            if timeout. Access field values via .value property or magic conversion.
        
        Examples
        --------
        Non-blocking read:
        
        >>> data = shm.read(timeout=0)
        >>> if data:
        ...     temp = data.temperature
        ...     if temp.valid and temp.modified:
        ...         print(f"New temp: {temp.value}")
        
        Blocking read with timeout:
        
        >>> data = fifo.read(timeout=1.0)
        >>> if data is None:
        ...     print("Timeout")
        
        Skip to most recent (FIFO):
        
        >>> data = fifo.read(timeout=0.5, latest=True)
        
        Single reader with change tracking:
        
        >>> data = shm.read(reset_modified=True)
        >>> if data.position.modified:
        ...     print("Position changed since last read")
        
        Notes
        -----
        - Returns None if no data available within timeout
        - Sequence numbers ensure consistent data (no partial writes)
        - FIFO with latest=True discards unread older data
        - reset_modified only works in single-slot mode
        - All field values are ValueWithStatus objects
        """
        if reset_modified and self.is_fifo:
            raise ValueError("reset_modified only supported in single-slot mode")
        
        if self.is_fifo:
            return self._read_fifo(timeout, latest)
        else:
            return self._read_single(timeout, reset_modified)
    
    def close(self):
        """
        Close shared memory connection.
        
        Must be called by all processes before exit. Does not delete the shared
        memory segment (use unlink() for that).
        
        Examples
        --------
        >>> shm = SharedMemory(MyData, name="test")
        >>> # ... use shm ...
        >>> shm.close()
        """
        self.shm.close()
    
    def unlink(self):
        """
        Delete shared memory segment from system.
        
        Should only be called by the process that created the shared memory
        (create=True). Call after all processes have called close().
        
        Examples
        --------
        >>> # In creator process:
        >>> shm = SharedMemory(MyData, name="test", create=True)
        >>> # ... use shm ...
        >>> shm.close()
        >>> shm.unlink()  # Delete from system
        """
        self.shm.unlink()
    
    def _initialize_slot(self, slot_idx: int):
        """Initialize slot with unwritten flags."""
        offset = self._get_slot_offset(slot_idx)
        
        # seq_begin = 0
        self._write_uint64(offset, 0)
        
        # All fields unwritten
        num_fields = len(self._layout.fields)
        status_offset = offset + 8
        for i in range(num_fields):
            self.shm.buf[status_offset + i] = FieldStatus.MASK_UNWRITTEN
        
        # seq_end = 0
        seq_end_offset = offset + self._slot_size - 8
        self._write_uint64(seq_end_offset, 0)
    
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
    
    def _write_to_slot(self, slot_idx: int, data: dict):
        """Write data to slot with sequence numbers."""
        offset = self._get_slot_offset(slot_idx)
        
        # Read current sequence
        seq = self._read_uint64(offset)
        
        # Increment sequence and write seq_begin
        seq += 1
        self._write_uint64(offset, seq)
        
        # Get status array offset
        status_offset = offset + 8
        
        # Write fields and update status
        for idx, field_info in enumerate(self._layout.fields):
            field_offset = offset + field_info.offset
            
            if field_info.name in data:
                # Field is being written
                value = data[field_info.name]
                
                # Write value
                truncated = False
                if field_info.is_scalar:
                    self._write_scalar(field_offset, value, field_info.field_type)
                elif field_info.is_string:
                    truncated = self._write_string(field_offset, value, field_info)
                elif field_info.is_array:
                    truncated = self._write_array(field_offset, value, field_info)
                
                # Update status: clear unwritten, set modified
                status = self.shm.buf[status_offset + idx]
                status &= ~FieldStatus.MASK_UNWRITTEN
                status |= FieldStatus.MASK_MODIFIED
                
                if truncated:
                    status |= FieldStatus.MASK_TRUNCATED
                else:
                    status &= ~FieldStatus.MASK_TRUNCATED
                
                self.shm.buf[status_offset + idx] = status
            else:
                # Field not written in this update - clear modified flag
                status = self.shm.buf[status_offset + idx]
                status &= ~FieldStatus.MASK_MODIFIED
                self.shm.buf[status_offset + idx] = status
        
        # Write seq_end (same as seq_begin = consistent)
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
        # Truncate by character count
        truncated = len(value) > field_info.string_max_chars
        if truncated:
            value = value[:field_info.string_max_chars]
        
        # Encode to UTF-8
        encoded = value.encode('utf-8')
        
        # Write length
        self._write_uint32(offset, len(encoded))
        
        # Write UTF-8 data
        self.shm.buf[offset+4:offset+4+len(encoded)] = encoded
        
        return truncated
    
    def _write_array(self, offset: int, value: np.ndarray, field_info: '_FieldInfo') -> bool:
        """Write array, return True if truncated."""
        # Convert to correct dtype
        value = np.asarray(value, dtype=field_info.array_dtype)
        
        # Check if shape matches
        truncated = value.shape != field_info.array_shape
        
        # Flatten and truncate if necessary
        flat_value = value.flatten()
        expected_size = int(np.prod(field_info.array_shape))
        
        if len(flat_value) > expected_size:
            flat_value = flat_value[:expected_size]
            truncated = True
        elif len(flat_value) < expected_size:
            # Pad with zeros
            flat_value = np.pad(flat_value, (0, expected_size - len(flat_value)))
        
        # Write to shared memory
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
            # Try to read with sequence check
            data = self._read_from_slot(0, reset_modified)
            if data is not None:
                return data
            
            # Check timeout
            if timeout == 0 or (time.time() - start_time) >= timeout:
                return None
            
            time.sleep(0.0001)  # 100 microseconds
    
    def _read_fifo(self, timeout: float, latest: bool) -> Optional[Any]:
        """Read from FIFO."""
        start_time = time.time()
        
        while True:
            write_idx, read_idx, count = self._get_fifo_metadata()
            
            # Check if data available
            if count == 0:
                if timeout == 0 or (time.time() - start_time) >= timeout:
                    return None
                time.sleep(0.0001)
                continue
            
            # Determine which slot to read
            if latest and count > 1:
                # Skip to most recent, discard older
                read_idx = write_idx - 1
                count = 1
            
            slot_idx = read_idx % self.slots
            data = self._read_from_slot(slot_idx, False)
            
            if data is None:
                # Retry
                time.sleep(0.0001)
                continue
            
            # Advance read pointer
            read_idx += 1
            count -= 1
            self._set_fifo_metadata(write_idx, read_idx, count)
            
            return data
    
    def _read_from_slot(self, slot_idx: int, reset_modified: bool) -> Optional[Any]:
        """Read from slot with sequence check."""
        offset = self._get_slot_offset(slot_idx)
        
        # Read seq_begin
        seq_begin = self._read_uint64(offset)
        
        # Get status array offset
        status_offset = offset + 8
        
        # Read all fields and status
        field_values = {}
        for idx, field_info in enumerate(self._layout.fields):
            field_offset = offset + field_info.offset
            
            # Read status byte
            status_byte = self.shm.buf[status_offset + idx]
            status = FieldStatus(status_byte)
            
            # Read value
            if field_info.is_scalar:
                value = self._read_scalar(field_offset, field_info.field_type)
            elif field_info.is_string:
                value = self._read_string(field_offset, field_info)
            elif field_info.is_array:
                value = self._read_array(field_offset, field_info)
            
            # Wrap in ValueWithStatus
            field_values[field_info.name] = ValueWithStatus(value, status)
        
        # Read seq_end
        seq_end_offset = offset + self._slot_size - 8
        seq_end = self._read_uint64(seq_end_offset)
        
        # Check consistency
        if seq_begin != seq_end:
            return None
        
        # Reset modified flags if requested
        if reset_modified:
            for idx in range(len(self._layout.fields)):
                status_byte = self.shm.buf[status_offset + idx]
                status_byte &= ~FieldStatus.MASK_MODIFIED
                self.shm.buf[status_offset + idx] = status_byte
        
        # Create dataclass instance
        return self.dataclass_type(**field_values)
    
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
        # Read length
        length = self._read_uint32(offset)
        
        # Read UTF-8 bytes
        encoded = bytes(self.shm.buf[offset+4:offset+4+length])
        
        # Decode
        return encoded.decode('utf-8', errors='ignore')
    
    def _read_array(self, offset: int, field_info: '_FieldInfo') -> np.ndarray:
        """Read array."""
        size = int(np.prod(field_info.array_shape))
        flat_array = np.ndarray(
            size,
            dtype=field_info.array_dtype,
            buffer=self.shm.buf,
            offset=offset
        ).copy()  # Copy to avoid shared memory reference
        
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
        
        # Header: seq_begin (8) + status_bytes (num_fields)
        offset = 8 + num_fields
        
        # Align to 8 bytes
        offset = (offset + 7) // 8 * 8
        
        # Fields
        for field_info in self.fields:
            field_info.offset = offset
            offset += field_info.size
        
        # Footer: seq_end (8)
        offset += 8
        
        # Align to 8 bytes
        self.total_size = (offset + 7) // 8 * 8


class _FieldInfo:
    """Information about a dataclass field."""
    
    def __init__(self, name: str, field_type: Any, default: Any):
        self.name = name
        self.field_type = field_type
        self.default = default
        self.is_scalar = False
        self.is_string = False
        self.is_array = False
        self.string_max_chars = 0
        self.array_dtype = None
        self.array_shape = None
        self.size = 0
        self.offset = 0
        
        self._parse_type()
    
    def _parse_type(self):
        """Determine field properties."""
        annotation = str(self.field_type)
        
        # Check for string
        string_chars = _AnnotationParser.parse_string(annotation)
        if string_chars:
            self.is_string = True
            self.string_max_chars = string_chars
            # 4 bytes for length + max_chars * 4 bytes for UTF-8
            self.size = 4 + string_chars * 4
            return
        
        # Check for array
        array_info = _AnnotationParser.parse_array(annotation)
        if array_info:
            self.is_array = True
            self.array_dtype, self.array_shape = array_info
            self.size = int(np.prod(self.array_shape)) * np.dtype(self.array_dtype).itemsize
            return
        
        # Scalar types
        type_sizes = {
            float: 8,  # float64
            int: 4,    # int32
            bool: 1,
        }
        
        if self.field_type in type_sizes:
            self.is_scalar = True
            self.size = type_sizes[self.field_type]
            return
        
        raise ValueError(f"Unsupported type for field '{self.name}': {self.field_type}")


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
        
        # Parse dtype
        dtype_map = {
            'float32': np.float32,
            'float64': np.float64,
            'int8': np.int8,
            'int16': np.int16,
            'int32': np.int32,
            'int64': np.int64,
            'uint8': np.uint8,
            'uint16': np.uint16,
            'uint32': np.uint32,
            'uint64': np.uint64,
            'bool': np.bool_,
        }
        dtype = dtype_map.get(dtype_str, np.float64)
        
        # Parse shape
        shape = tuple(int(x) for x in shape_str.split(','))
        
        return dtype, shape


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    from multiprocessing import Process
    import time
    
    # Define status dataclass (no status fields needed!)
    @dataclass
    class ControllerStatus:
        position: float = 0.0
        velocity: float = 0.0
        target: float = 0.0
        error: float = 0.0
        timestamp: float = 0.0
        mode: int = 0
        is_active: bool = False
        message: "str[64]" = ""
        image: "float32[10,10]" = None
    
    # === SINGLE-SLOT EXAMPLE ===
    def single_slot_writer():
        shm = SharedMemory(ControllerStatus, name="single_test", create=True, slots=1)
        
        for i in range(20):
            shm.write(
                position=i * 0.1,
                velocity=i * 0.05,
                timestamp=time.time(),
                is_active=True,
                message=f"Iteration {i}",
                image=np.random.rand(10, 10).astype(np.float32)
            )
            time.sleep(0.1)
        
        shm.close()
        shm.unlink()
    
    def single_slot_reader():
        time.sleep(0.05)
        shm = SharedMemory(ControllerStatus, name="single_test", create=False, slots=1)
        
        for _ in range(20):
            data = shm.read(timeout=1.0, reset_modified=True)
            if data:
                pos = data.position
                msg = data.message
                print(f"Single: pos={pos.value:.2f}, msg={msg.value}, "
                      f"pos_modified={pos.modified}, msg_valid={msg.valid}")
            time.sleep(0.1)
        
        shm.close()
    
    # === FIFO EXAMPLE ===
    def fifo_writer():
        fifo = SharedMemory(ControllerStatus, name="fifo_test", create=True, slots=5)
        
        for i in range(30):
            fifo.write(
                position=i * 0.1,
                velocity=i * 0.05,
                timestamp=time.time(),
                message=f"FIFO {i}"
            )
            fifo.finalize()
            time.sleep(0.05)
        
        fifo.close()
        fifo.unlink()
    
    def fifo_reader():
        time.sleep(0.1)
        fifo = SharedMemory(ControllerStatus, name="fifo_test", create=False, slots=5)
        
        # Read all
        print("\n=== Reading all ===")
        for _ in range(15):
            data = fifo.read(timeout=0.5, latest=False)
            if data:
                pos = data.position
                msg = data.message
                print(f"FIFO: pos={pos.value:.2f}, msg={msg.value}, modified={pos.modified}")
            time.sleep(0.15)
        
        # Read only latest
        print("\n=== Reading latest only ===")
        for _ in range(5):
            data = fifo.read(timeout=0.5, latest=True)
            if data:
                pos = data.position
                print(f"Latest: pos={pos.value:.2f}, modified={pos.modified}")
            time.sleep(0.2)
        
        fifo.close()
    
    # Run examples
    print("=== SINGLE-SLOT EXAMPLE ===")
    p1 = Process(target=single_slot_writer)
    p2 = Process(target=single_slot_reader)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
    print("\n\n=== FIFO EXAMPLE ===")
    p3 = Process(target=fifo_writer)
    p4 = Process(target=fifo_reader)
    p3.start()
    p4.start()
    p3.join()
    p4.join()