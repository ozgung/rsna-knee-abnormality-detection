"""Ordered single-producer prefetch; checkpoints use acknowledged consumption only.

Callbacks must not mutate global RNG and must return the existing volume/mask
objects unchanged. No decoding, normalization, sampling, or replacement occurs here.
A running callback cannot be forcibly cancelled: close fails visibly on its timeout.
"""
from collections import OrderedDict, deque
from threading import Condition, Thread


class StudyLRU:
    """Optional study-aligned cache, with explicit immutable input-contract identity."""
    def __init__(self, capacity=64):
        if not isinstance(capacity, int) or not 0 <= capacity <= 64:
            raise ValueError('Cache capacity must be between zero and 64')
        self.capacity = capacity
        self._items = OrderedDict()

    def read(self, contract, study, callback):
        if not isinstance(contract, str) or not contract:
            raise ValueError('Nonempty immutable contract identity required')
        key = (contract, study)
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        value = callback(study)
        if self.capacity:
            self._items[key] = value
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return value


class StudyStream:
    """One outstanding consumer delivery; acknowledge only after actual consumption.

    Reconstruct with consumed_cursor to resume. Pending deliveries and fetched-ahead
    buffers are deliberately absent from the checkpoint. LRU is single-producer
    owned; reuse it only after closing the preceding stream.
    """
    def __init__(self, studies, read, *, contract, consumed_cursor=0,
                 prefetch=4, cache=None, close_timeout=5.0):
        self.studies = tuple(studies)
        if (not isinstance(consumed_cursor, int) or
                not 0 <= consumed_cursor <= len(self.studies)):
            raise ValueError('Invalid consumed cursor')
        if not isinstance(prefetch, int) or not 1 <= prefetch <= 4:
            raise ValueError('Prefetch must be between one and four')
        if close_timeout <= 0:
            raise ValueError('Positive close timeout required')
        if not isinstance(contract, str) or not contract:
            raise ValueError('Nonempty immutable contract identity required')
        self.consumed_cursor = consumed_cursor
        self.fetched_cursor = consumed_cursor
        self._next = consumed_cursor
        self._pending = None
        self._buffer = deque()
        self._condition = Condition()
        self._closed = False
        self._error = None
        self._done = False
        self._prefetch = prefetch
        self._timeout = close_timeout
        self._cache = cache if cache is not None else StudyLRU(0)
        self._read = read
        self._contract = contract
        self._thread = Thread(target=self._produce, name='public-study-producer', daemon=True)
        self._thread.start()

    def _produce(self):
        try:
            for index in range(self._next, len(self.studies)):
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or len(self._buffer) < self._prefetch)
                    if self._closed:
                        return
                value = self._cache.read(self._contract, self.studies[index], self._read)
                with self._condition:
                    if self._closed:
                        return
                    self._buffer.append((index, value))
                    self.fetched_cursor = index + 1
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
        finally:
            with self._condition:
                self._done = True
                self._condition.notify_all()

    def __iter__(self):
        return self

    def __next__(self):
        with self._condition:
            if self._closed:
                raise RuntimeError('Stream is closed')
            if self._pending is not None:
                raise RuntimeError('Previous delivery requires acknowledgment')
            self._condition.wait_for(lambda: self._closed or self._buffer or self._done)
            if self._closed:
                raise RuntimeError('Stream is closed')
            if self._buffer:
                index, value = self._buffer.popleft()
                self._pending = index
                self._condition.notify_all()
                return value
            if self._error is not None:
                raise self._error
            raise StopIteration

    def acknowledge(self):
        with self._condition:
            if self._closed or self._pending is None:
                raise RuntimeError('No open delivery to acknowledge')
            if self._pending != self.consumed_cursor:
                raise RuntimeError('Consumption order changed')
            self.consumed_cursor += 1
            self._pending = None

    def close(self):
        with self._condition:
            self._closed = True
            self._buffer.clear()
            self._pending = None
            self._condition.notify_all()
        self._thread.join(self._timeout)
        if self._thread.is_alive():
            raise RuntimeError('Study callback did not finish during close timeout')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
