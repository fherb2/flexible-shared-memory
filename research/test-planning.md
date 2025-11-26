# Test Planning for Flexible Shared Memory

This document outlines the comprehensive test strategy for achieving production-ready test coverage suitable for PyPI publication.

## Test Philosophy

* **Isolation** : Each test should be independent and not affect others
* **Reproducibility** : Tests must be deterministic and repeatable
* **Coverage** : Aim for >90% code coverage with meaningful tests
* **Edge Cases** : Explicitly test boundary conditions and error cases
* **Multi-Process** : Test real inter-process communication scenarios
* **Performance** : Validate lock-free behavior and performance characteristics

## Test Structure Overview

Tests are organized into focused modules:

1. **test_write_basic.py** - Writing data to shared memory
2. **test_read_basic.py** - Reading data from shared memory
3. **test_field_status.py** - Field status flag behavior
4. **test_value_conversion.py** - ValueWithStatus wrapper functionality
5. **test_types.py** - Type handling and parsing
6. **test_fifo.py** - FIFO mode operations
7. **test_multiprocess.py** - Inter-process communication
8. **test_edge_cases.py** - Boundary conditions and error handling
9. **test_performance.py** - Performance and concurrency validation

---

## 1. test_write_basic.py

### Priority 1 (Implement First)

* ✅ Write single scalar field (float, int, bool)
* ✅ Write multiple scalar fields in one call
* ✅ Write to single-slot mode
* ✅ Write string field (UTF-8)
* ✅ Write array field (NumPy)
* ✅ Verify sequence numbers increment
* ✅ Test write() updates modified flag

### Priority 2 (Detailed Coverage)

* Write partial updates (only some fields)
* Write same field multiple times
* Write with empty dataclass
* Write with all default values
* Verify unwritten fields remain unwritten
* Test write() atomicity with sequence numbers

### Priority 3 (Edge Cases)

* Write extremely large values (near limits)
* Write zero values
* Write negative values
* Write NaN/Inf for floats
* Write unicode edge cases (emoji, RTL text, combining chars)
* Write to full memory (test memory layout)

---

## 2. test_read_basic.py

### Priority 1 (Implement First)

* ✅ Read single-slot after write
* ✅ Read returns None with timeout=0 when empty
* ✅ Read blocks until data available
* ✅ Read returns correct scalar values
* ✅ Read returns correct string values
* ✅ Read returns correct array values
* ✅ Verify sequence number consistency check works

### Priority 2 (Detailed Coverage)

* Read timeout behavior (timeout=0, timeout>0)
* Read multiple times from same slot
* Read with reset_modified=True
* Read with reset_modified=False
* Read immediately after write (no delay)
* Read after multiple writes to same slot
* Verify read doesn't modify data

### Priority 3 (Edge Cases)

* Read from uninitialized slot
* Read during concurrent write (sequence mismatch retry)
* Read timeout precision
* Read with very short timeout (< 1ms)
* Read with very long timeout (> 60s)

---

## 3. test_field_status.py

### Priority 1 (Implement First)

* ✅ Test valid flag for written fields (exact copy only)
* ✅ Test unwritten flag on new slot
* ✅ Test modified flag set on write
* ✅ Test truncated flag for oversized string
* ✅ Test truncated flag for oversized array
* ✅ Verify flags persist across reads
* ✅ Test that truncated fields are NOT valid (data may be unusable)

### Priority 2 (Detailed Coverage)

* Test unwritten cleared after first write
* Test modified reset in FIFO new slot
* Test modified reset with reset_modified=True
* Test truncated flag only for affected fields
* Test multiple status flags simultaneously (modified + truncated)
* Verify status independent per field
* Test FieldStatus bit masks directly

### Priority 3 (Edge Cases)

* Status after write with no actual change in value
* Status consistency across multiple readers
* Status flag behavior during sequence mismatch
* Status with partial field updates
* Reserved bits remain zero

---

## 4. test_value_conversion.py

### Priority 1 (Implement First)

* ✅ Test .value property access
* ✅ Test .valid property
* ✅ Test .modified property
* ✅ Test .truncated property
* ✅ Test .unwritten property
* ✅ Test float() conversion
* ✅ Test int() conversion
* ✅ Test str() conversion
* ✅ Test np.array() conversion for arrays

### Priority 2 (Detailed Coverage)

* Test bool() conversion
* Test arithmetic operations (+, -, *, /)
* Test comparison operations
* Test  **repr** () output format
* Test ValueWithStatus with None value
* Test ValueWithStatus with complex types
* Verify magic methods preserve value accuracy

### Priority 3 (Edge Cases)

* Conversion type errors (int from string)
* Conversion overflow (large float to int)
* Array conversion with wrong shape
* Chained operations (pos + 1) * 2
* Mixed type arithmetic (ValueWithStatus + native)

---

## 5. test_types.py

### Priority 1 (Implement First)

* ✅ Parse "str[64]" annotation
* ✅ Parse "float32[10,20]" annotation
* ✅ Parse scalar types (float, int, bool)
* ✅ Verify string memory allocation (4 + chars*4)
* ✅ Test UTF-8 string round-trip

### Priority 2 (Detailed Coverage)

* Parse all supported NumPy dtypes
* Parse multi-dimensional arrays [10,20,3]
* Parse single-dimensional arrays [100]
* Test string length limits (1, 64, 256 chars)
* Test all scalar types in one dataclass
* Verify memory layout calculations
* Test field offset calculations

### Priority 3 (Edge Cases)

* Invalid annotations (malformed strings)
* Unsupported types (dict, list, custom class)
* Very large strings (str[10000])
* Very large arrays (float64[1000,1000,100])
* Zero-length arrays
* Mixed type dataclass with 20+ fields
* Unicode strings: Chinese, Japanese, Arabic, Emoji
* String truncation at UTF-8 boundaries

---

## 6. test_fifo.py

### Priority 1 (Implement First)

* ✅ Write and finalize to FIFO
* ✅ Read from FIFO in order
* ✅ Test FIFO with 2, 5, 10 slots
* ✅ Test FIFO overflow (overwrite oldest)
* ✅ Test latest=True skips to newest
* ✅ Test latest=False reads in order
* ✅ Verify finalize() required for FIFO

### Priority 2 (Detailed Coverage)

* Test write() without finalize() (data not visible)
* Test multiple write() before finalize() (staged)
* Test FIFO empty returns None
* Test FIFO metadata (write_idx, read_idx, count)
* Test modified flag behavior across slots
* Test partial field updates in FIFO
* Verify slot isolation (data doesn't leak)

### Priority 3 (Edge Cases)

* FIFO with slots=1 (edge case, should work)
* Write faster than read (continuous overflow)
* Read faster than write (frequent empty)
* Finalize() called twice without write()
* Read with latest=True when count=1
* FIFO wrap-around (indices > slots)
* Concurrent writers to FIFO (should work)

---

## 7. test_multiprocess.py

### Priority 1 (Implement First)

* ✅ One writer, one reader process
* ✅ Writer creates, reader opens with create=False
* ✅ Data transfer correctness
* ✅ Reader sees modified flags
* ✅ Test with realistic dataclass (sensor data)

### Priority 2 (Detailed Coverage)

* One writer, multiple readers
* Multiple writers, one reader (sequential)
* FIFO with multiple readers
* Process cleanup (close/unlink)
* Test with Process and with Pool
* Verify no data corruption
* Test large data transfer (images)

### Priority 3 (Edge Cases)

* Reader starts before writer creates memory
* Writer unlinks while reader active (error handling)
* Very fast write/read cycles (stress test)
* Process crash handling (cleanup)
* 10+ concurrent readers
* Shared memory persistence after writer exits
* Name collision (two SharedMemory with same name)

---

## 8. test_edge_cases.py

### Priority 1 (Implement First)

* ✅ Empty dataclass (no fields)
* ✅ Dataclass with only one field
* ✅ Very large dataclass (50+ fields)
* ✅ String overflow by 1 character
* ✅ String overflow by many characters
* ✅ Array shape mismatch (wrong dimensions)

### Priority 2 (Detailed Coverage)

* Zero-length string write
* Empty array write (shape with 0)
* Write None to optional fields
* Sequence number overflow (2^64)
* Memory alignment edge cases
* Slot size edge cases (very small/large)
* Unicode combining characters in strings
* Right-to-left text (Arabic, Hebrew)

### Priority 3 (Comprehensive Edge Cases)

* System-generated names (no collision)
* Very long custom names (>100 chars)
* Special characters in names
* Create=False before creator
* Double close() call
* Double unlink() call
* Read/write after close()
* Timeout with negative value
* Timeout with infinity
* Reset_modified in FIFO mode (should raise)
* Finalize in single-slot mode (should raise)

---

## 9. test_performance.py

### Priority 2 (Performance Validation)

* Measure single-slot write latency
* Measure single-slot read latency
* Measure FIFO throughput
* Test 1000+ writes/sec sustained
* Verify lock-free (no blocking on read)
* Test with large arrays (1MB+)
* Memory usage validation
* Sequence number retry rate (should be very low)

### Priority 3 (Stress Tests)

* 10,000 writes without read
* 100,000 read/write cycles
* Concurrent readers (10+)
* Maximum supported field count
* Maximum string length
* Maximum array size
* Memory leak check (repeated create/destroy)

---

## Test Utilities and Fixtures

### Fixtures Needed

* `temp_shared_memory()` - Auto-cleanup after test
* `sample_dataclasses()` - Various test dataclasses
* `multiprocess_context()` - Setup/teardown for processes
* `random_data_generator()` - Generate test data

### Helper Functions

* `assert_field_equal(actual, expected)` - Compare with tolerance
* `wait_for_condition(condition, timeout)` - Wait helper
* `measure_latency(operation)` - Performance helper
* `verify_memory_layout(dataclass)` - Layout validation

---

## Test Execution Strategy

### Phase 1: Basic Functionality (Current Implementation)

Implement Priority 1 tests from:

* test_write_basic.py
* test_read_basic.py
* test_field_status.py
* test_value_conversion.py
* test_fifo.py
* test_multiprocess.py

 **Goal** : Verify core functionality works correctly

### Phase 2: Detailed Coverage

Implement Priority 2 tests from all modules

 **Goal** : Achieve >85% code coverage

### Phase 3: Edge Cases and Stress Tests

Implement Priority 3 tests + test_performance.py

 **Goal** : Production-ready, >95% coverage, validated edge cases

---

## Continuous Integration

### GitHub Actions Matrix

* Python: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13
* OS: Ubuntu Latest
* NumPy: Minimum supported (1.20) and latest

### CI Checks

* All tests pass
* Code coverage >85% (Phase 2)
* No warnings during test execution
* Linting (optional: black, flake8)

### Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=flexible_shared_memory --cov-report=html

# Run specific test file
pytest tests/test_write_basic.py -v

# Run single test
pytest tests/test_write_basic.py::test_write_single_float -v
```

---

## Notes for Test Implementation

1. **Cleanup** : Every test must cleanup shared memory (use fixtures)
2. **Isolation** : Use unique names for each test's shared memory
3. **Timing** : Use `time.sleep()` judiciously for multi-process tests
4. **Assertions** : Use descriptive assertion messages
5. **Documentation** : Each test should have docstring explaining what it validates
6. **Reproducibility** : Avoid randomness without fixed seeds

---

## Test Metrics Goals

| Metric        | Phase 1 | Phase 2       | Phase 3    |
| ------------- | ------- | ------------- | ---------- |
| Code Coverage | >70%    | >85%          | >95%       |
| Test Count    | ~50     | ~150          | ~300       |
| Edge Cases    | Basic   | Comprehensive | Exhaustive |
| Performance   | None    | Basic         | Detailed   |

---

## Future Test Considerations

* **Integration tests** : Test with real applications (camera capture, sensor logging)
* **Platform tests** : Windows, macOS (Phase 4)
* **Python implementations** : PyPy, Pyston (optional)
* **Benchmarking** : Compare with alternatives (multiprocessing.Queue, etc.)
* **Documentation tests** : Verify all examples in README work
