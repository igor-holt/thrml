import pytest
from thrml.pgm import _CounterMeta
import threading
import time

def test_class_ordering_uniqueness():
    """Verify that classes created with _CounterMeta have a strict, unique ordering."""

    # Helper to create dynamic classes
    def make_class(name):
        return _CounterMeta(name, (), {})

    # Create two classes with identical module and qualname
    C1 = make_class("C")
    C2 = make_class("C")

    # Ensure they are distinct objects
    assert C1 is not C2
    assert C1 != C2

    # Ensure they have identical string representations
    assert C1.__module__ == C2.__module__
    assert C1.__qualname__ == C2.__qualname__

    # Check strict ordering (one must be less than the other)
    is_c1_lt_c2 = C1 < C2
    is_c2_lt_c1 = C2 < C1

    assert is_c1_lt_c2 != is_c2_lt_c1, "Classes must be strictly ordered"
    assert (is_c1_lt_c2 or is_c2_lt_c1), "One class must be less than the other"

    # Check stability of sorting
    keys = [C1, C2]
    sorted_keys_1 = sorted(keys)
    keys_reversed = [C2, C1]
    sorted_keys_2 = sorted(keys_reversed)

    assert sorted_keys_1 == sorted_keys_2, "Sorting should be deterministic regardless of input order"

    # Verify _class_id attribute exists
    assert hasattr(C1, "_class_id")
    assert hasattr(C2, "_class_id")
    assert C1._class_id != C2._class_id

def test_ordering_with_different_classes():
    """Verify ordering still works for different classes."""
    def make_class(name):
        return _CounterMeta(name, (), {})

    A = make_class("A")
    B = make_class("B")

    # Comparison should prioritize name/module
    # 'A' < 'B' alphabetically
    # Assuming module is same
    assert A < B
    assert not (B < A)

def test_thread_safe_class_creation():
    """Verify that class creation is thread-safe and produces unique _class_id values."""
    num_threads = 10
    classes_per_thread = 10
    all_classes = []
    lock = threading.Lock()
    
    def create_classes(thread_id):
        """Create classes in a separate thread."""
        local_classes = []
        for i in range(classes_per_thread):
            cls = _CounterMeta(f"ThreadClass_{thread_id}_{i}", (), {})
            local_classes.append(cls)
            # Small sleep to increase likelihood of race conditions if lock is missing
            time.sleep(0.0001)
        
        with lock:
            all_classes.extend(local_classes)
    
    # Create and start threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=create_classes, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify we created the expected number of classes
    assert len(all_classes) == num_threads * classes_per_thread
    
    # Verify all _class_id values are unique
    class_ids = [cls._class_id for cls in all_classes]
    assert len(class_ids) == len(set(class_ids)), "All _class_id values must be unique"
    
    # Verify all classes have valid _class_id attributes
    for cls in all_classes:
        assert hasattr(cls, "_class_id")
        assert isinstance(cls._class_id, int)
        assert cls._class_id >= 0
