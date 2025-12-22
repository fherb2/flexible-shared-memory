# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.3.0] - 2024-12-22

### Added
- **Auto-generated shared memory names**: Writers no longer require manual name specification
  - Names are automatically generated using UUID (format: `shm_<8-char-hex>`)
  - Names are accessible via `shm.name` property after creation
- **`inspect()` class method**: Inspect shared memory structure without attaching
  - Returns `SHMInspectionInfo` with slots, size, and field details
  - Useful for debugging and validation
- **`expected_type` parameter**: Validate dataclass structure when attaching to existing shared memory
  - Compares hash of expected structure with stored structure
  - Raises clear error message on mismatch
- **Automatic DataClass reconstruction**: Readers can attach without importing the original DataClass
  - DataClass is reconstructed from shared memory header metadata
  - Field names, types, and layout are preserved exactly

### Changed
- **API Simplification**: Writer no longer needs `name` parameter (breaking change for code that relied on custom names)
- **Improved error messages**: Structure mismatch errors now show expected vs found hash values
- **Python version requirement**: Now requires Python 3.7+ (stable field ordering guarantee)

### Fixed
- Security: Header validation now uses hash comparison instead of direct structure comparison
- Removed potential race condition in header parsing

### Breaking Changes
- `SharedMemory(DataClass, name="custom_name")` is no longer supported
  - Use auto-generated names: `shm = SharedMemory(DataClass); name = shm.name`
  - For compatibility with existing code expecting custom names, manually track the mapping

### Migration Guide
**Before (0.2.0):**
```python
# Writer
shm = SharedMemory(SensorData, name="sensor_buffer", slots=5)
pipe.send("sensor_buffer")
```

**After (0.3.0):**
```python
# Writer
shm = SharedMemory(SensorData, slots=5)
pipe.send(shm.name)  # Auto-generated name
```


## [0.2.0] - 2024-12-21

### Added
- Self-describing header with automatic configuration detection
- Readers no longer need to specify `slots` parameter (auto-detected from header)
- Hash-based header validation prevents dataclass mismatches across processes
- `overflow` flag for FIFO mode - indicates when older data was lost due to buffer overflow
- Comprehensive test suite with fork/spawn coverage (~96% code coverage, 200+ tests)

### Changed
- **Performance**: Object pooling reduces allocations by ~83% (6x improvement in read-heavy workloads)
- **Performance**: Pre-cached field metadata eliminates repeated calculations
- String truncation now happens on character boundary before UTF-8 encoding (more predictable)
- Improved error messages for dataclass structure mismatches

### Fixed
- UTF-8 string truncation now correctly truncates by character count, not byte count
- Security: Removed `eval()` usage in field deserialization (CVE prevention)
- Field status tracking now correctly uses object pooling for lock-free performance
- Header hash validation now correctly compares reader's layout with stored layout

### Internal
- Added `string_max_bytes` and `array_flat_size` caching for performance
- `_read_dict`, `_write_buffer`, `_field_status_pool`, `_value_status_pool` now reused
- `FieldStatus` and `ValueWithStatus` objects pooled to avoid allocations

## [0.1.0] - 2024-12-20

### Added
- Initial release
- Lock-free shared memory with sequence numbers
- Field-level status tracking (valid, modified, truncated, unwritten)
- Support for scalar types (float, int, bool)
- UTF-8 string support with character limits
- NumPy array support (1D, 2D, 3D+)
- Single-slot and FIFO modes
- Pythonic ValueWithStatus wrapper with magic methods
