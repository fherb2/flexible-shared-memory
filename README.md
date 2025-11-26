# Flexible Shared Memory

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
# Create shared memory
shm = SharedMemory(SensorData, name="sensors")

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
# Connect to existing shared memory
shm = SharedMemory(SensorData, name="sensors", create=False)

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

```python
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
  
if pos.modified:        # Changed since last slot/read
    print("Position updated!")
```

 **Important** : A field is only `valid` if it contains the complete, exact data from the source. Truncated data (e.g., incomplete array coefficients) may be completely unusable and should be treated as an error condition.

## FIFO Mode

For buffered communication with multiple slots:

```python
# Create FIFO with 10 slots
fifo = SharedMemory(SensorData, name="buffer", slots=10)

# Writer: Stage and commit
fifo.write(temperature=23.5)
fifo.write(pressure=1013.0)
fifo.finalize()  # Atomic commit

# Reader: Read oldest or skip to latest
data = fifo.read(timeout=1.0, latest=False)  # FIFO order
data = fifo.read(timeout=0.5, latest=True)   # Skip to newest
```

## Advanced Features

### Practical Error Handling

```python
# Example: Polynomial coefficients
@dataclass
class MathData:
    coefficients: "float64[5]" = None  # [a0, a1, a2, a3, a4]

shm = SharedMemory(MathData, name="math")
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
