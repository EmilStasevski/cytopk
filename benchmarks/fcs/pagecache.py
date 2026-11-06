"""Evict a file from the Linux page cache without root, and measure how much of it is resident.

drop(path)      posix_fadvise(POSIX_FADV_DONTNEED) on the whole file. Works for clean pages of files we can
                read; it does not touch the drive's own cache (no root, so no /proc/sys/vm/drop_caches).
resident(path)  fraction of the file's pages in the page cache (mincore on a read-only mapping).

    python pagecache.py drop <file>...      (used by flowcore_reader.R between cold reads)
"""

from __future__ import annotations

import ctypes
import os
import sys

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
_PROT_READ, _MAP_SHARED = 1, 1


def drop(path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def resident(path) -> float:
    size = os.path.getsize(path)
    if size == 0:
        return 0.0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap failed")
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            vec = ctypes.create_string_buffer((size + page - 1) // page)
            if _libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            return sum(b & 1 for b in vec.raw) / len(vec.raw)
        finally:
            _libc.munmap(addr, size)
    finally:
        os.close(fd)


if __name__ == "__main__":
    if sys.argv[1] == "drop":
        for p in sys.argv[2:]:
            drop(p)
