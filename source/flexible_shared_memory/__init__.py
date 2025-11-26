"""
Flexible Shared Memory
======================

A high-performance, lock-free shared memory system for inter-process 
communication in Python with automatic type mapping and field-level 
status tracking.

License
-------
MIT License

Copyright (c) 2024 fherb2

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Contact
-------
GitLab: https://gitlab.com/fherb2/flexible-shared-memory
Issues: https://gitlab.com/fherb2/flexible-shared-memory/-/issues
"""

from .shared_memory import (
    SharedMemory,
    FieldStatus,
    ValueWithStatus,
)

__version__ = "0.1.0"
__author__ = "fherb2"
__license__ = "MIT"

__all__ = [
    "SharedMemory",
    "FieldStatus",
    "ValueWithStatus",
]