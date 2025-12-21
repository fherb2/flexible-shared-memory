# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
