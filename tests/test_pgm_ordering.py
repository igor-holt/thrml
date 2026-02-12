import pytest
from thrml.pgm import _CounterMeta

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
