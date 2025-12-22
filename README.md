Flexible Shared Memory
Somwhere unpickled between multiprocessing Shared Memory and -Queue

[![Tests](https://github.com/fherb2/flexible-shared-memory/actions/workflows/test.yml/badge.svg)](https://github.com/fherb2/flexible-shared-memory/actions)

A high-performance, lock-free shared memory system for inter-process communication in Python with automatic type mapping and field-level status tracking.

## Features

* **Automatic type mapping** : Define data structures as dataclasses, shared memory layout is generated automatically
* **Lock-free operation** : Uses sequence numbers for consistent reads without locks
* **Field-level status tracking** : Each field tracks valid, modified, truncated, and unwritten state
* **NumPy integration** : Supports multi-dimensional arrays for images, sensor data, etc.
* **UTF-8 strings** : Full Unicode support with character-count based limits
* **Single-slot & FIFO modes** : Choose between minimal latency or buffered communication
* **Pythonic API** : Intuitive access to values and status flags

## Installation

```bash
pip install flexible-shared-memory
```

Or with poetry:

```bash
poetry add flexible-shared-memory
```

## Quick Start

### Define Your Data Structure

```python
from dataclasses import dataclass
from flexible_shared_memory import SharedMemory

@dataclass
class SensorData:
    temperature: float = 0.0
    pressure: float = 0.0
    timestamp: float = 0.0
    status_msg: "str[32]" = ""
```

### Writer Process

```python
# Create shared memory (name is auto-generated)
shm = SharedMemory(SensorData)

# The auto-generated name is available via shm.name
print(f"Created shared memory: {shm.name}")  # e.g., "shm_a3f8b2c1"

# Write data
shm.write(
    temperature=23.5,
    pressure=1013.25,
    timestamp=time.time(),
    status_msg="OK"
)

# Cleanup
shm.close()
shm.unlink()
```

### Reader Process

```python
# Connect to existing shared memory by name
# (DataClass is auto-reconstructed from header!)
shm = SharedMemory("shm_a3f8b2c1")

# Optional: Validate structure matches expected type
shm = SharedMemory("shm_a3f8b2c1", expected_type=SensorData)

# Read data with status
data = shm.read(timeout=1.0)

# Access values with status checking
temp = data.temperature
if temp.valid and temp.modified:
    print(f"Temperature: {temp.value}°C")

shm.close()
```

## Type Annotations

### Scalar Types

* `float` → 64-bit float
* `int` → 32-bit integer
* `bool` → boolean

### Strings (UTF-8)

* `"str[64]"` → Max 64 Unicode characters
* Supports all languages: ASCII, Cyrillic, Chinese, Japanese, Emojis

### Arrays (NumPy)

* `"float32[480,640,3]"` → 3D array (e.g., RGB image)
* Supported dtypes: `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bool`

## Field Status Tracking

Each field provides status information:

`````python
data = shm.read()
pos = data.position

# Check status
if pos.valid:           # Exact copy from source (not truncated/unwritten)
    use(pos.value)
else:
    # Data is NOT reliable - handle error
    if pos.truncated:
        log.error("Data truncated - may be unusable!")
    elif pos.unwritten:
        log.warning("No data available yet")
    elif pos.overflow:
        log.warning("FIFO overflow - older data was lost!")
  
if pos.modified:        # Changed since last slot/read
    print("Position updated!")
`````

**Important** : A field is only `valid` if it contains the complete, exact data from the source. Truncated data (e.g., incomplete array coefficients) may be completely unusable and should be treated as an error condition.

## FIFO Mode

For buffered communication with multiple slots:

```python
# Writer: Create FIFO with 10 slots (name auto-generated)
fifo = SharedMemory(SensorData, slots=10)
pipe.send(fifo.name)  # Send name to reader

# Writer: Stage and commit
fifo.write(temperature=23.5)
fifo.write(pressure=1013.0)
fifo.finalize()  # Atomic commit

# Reader: Attach by name (slots auto-detected from header)
name = pipe.recv()
reader = SharedMemory(name)

# Reader: Read oldest or skip to latest
data = reader.read(timeout=1.0, latest=False)  # FIFO order
data = reader.read(timeout=0.5, latest=True)   # Skip to newest
```

## Advanced Features

### Practical Error Handling

```python
# Example: Polynomial coefficients
@dataclass
class MathData:
    coefficients: "float64[5]" = None  # [a0, a1, a2, a3, a4]

shm = SharedMemory(MathData)
data = shm.read()

coeffs = data.coefficients
if coeffs.valid:
    # Safe to use - exact copy
    polynomial = np.poly1d(coeffs.value)
    result = polynomial(x)
else:
    if coeffs.truncated:
        # CRITICAL ERROR: Incomplete polynomial!
        # [a0, a1, a2] is a completely different polynomial than [a0, a1, a2, a3, a4]
        raise ValueError("Polynomial coefficients incomplete - cannot compute!")
    elif coeffs.unwritten:
        # Use default or skip computation
        pass
```

### Single-Reader Change Tracking

```python
# Reset modified flags after reading
data = shm.read(reset_modified=True)
if data.temperature.modified:
    print("Temperature changed since last read!")
```

### Magic Conversions

```python
pos = data.position
x = float(pos)      # Automatic conversion
y = pos.value       # Explicit access

# Works with NumPy arrays too
img = data.image
arr = np.array(img)  # Magic conversion
arr = img.value      # Explicit access
```

### Inspecting Shared Memory

Inspect shared memory structure without attaching:

```python
# Inspect without creating SharedMemory instance
info = SharedMemory.inspect("shm_a3f8b2c1")

print(f"Name: {info.name}")
print(f"Slots: {info.slots}")
print(f"Total size: {info.total_size} bytes")
print("\nFields:")
for field_name, field_type, field_size in info.fields:
    print(f"  {field_name}: {field_type} ({field_size} bytes)")
```

Output:

```
Name: shm_a3f8b2c1
Slots: 1
Total size: 128 bytes

Fields:
  temperature: float64 (8 bytes)
  pressure: float64 (8 bytes)
  timestamp: float64 (8 bytes)
  status_msg: str[32] (132 bytes)
```

This is useful for:

- Debugging shared memory structure
- Validating configuration before attaching
- Building monitoring tools

## Requirements

* Python ≥ 3.9
* NumPy ≥ 1.26 (supports NumPy 1.26+ and 2.x)

## License

MIT License - see LICENSE file for details

## Contributing

Issues and pull requests are welcome on GitLab:

https://gitlab.com/fherb2/flexible-shared-memory

## Contact

* GitLab: [@fherb2](https://gitlab.com/fherb2)
* Issues: https://gitlab.com/fherb2/flexible-shared-memory/-/issues

## Examples

See the `examples/` directory for complete usage examples including:

* Single-slot communication
* FIFO buffering
* Image transfer
* Multi-process applications
