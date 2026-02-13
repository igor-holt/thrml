import unittest
from thrml.pgm import _CounterMeta

class TestClassOrdering(unittest.TestCase):
    def test_class_ordering_uniqueness(self):
        """Verify that classes created with _CounterMeta have a strict, unique ordering."""

        # Helper to create dynamic classes
        def make_class(name):
            return _CounterMeta(name, (), {})

        # Create two classes with identical module and qualname
        C1 = make_class("C")
        C2 = make_class("C")

        # Ensure they are distinct objects
        self.assertIsNot(C1, C2)
        self.assertNotEqual(C1, C2)

        # Ensure they have identical string representations
        self.assertEqual(C1.__module__, C2.__module__)
        self.assertEqual(C1.__qualname__, C2.__qualname__)

        # Check strict ordering (one must be less than the other)
        is_c1_lt_c2 = C1 < C2
        is_c2_lt_c1 = C2 < C1

        self.assertNotEqual(is_c1_lt_c2, is_c2_lt_c1, "Classes must be strictly ordered")
        self.assertTrue(is_c1_lt_c2 or is_c2_lt_c1, "One class must be less than the other")

        # Check stability of sorting
        keys = [C1, C2]
        sorted_keys_1 = sorted(keys)
        keys_reversed = [C2, C1]
        sorted_keys_2 = sorted(keys_reversed)

        self.assertEqual(sorted_keys_1, sorted_keys_2, "Sorting should be deterministic regardless of input order")

        # Verify _class_id attribute exists
        self.assertTrue(hasattr(C1, "_class_id"))
        self.assertTrue(hasattr(C2, "_class_id"))
        self.assertNotEqual(C1._class_id, C2._class_id)

    def test_ordering_with_different_classes(self):
        """Verify ordering still works for different classes."""
        def make_class(name):
            return _CounterMeta(name, (), {})

        A = make_class("A")
        B = make_class("B")

        # Comparison should prioritize name/module
        # 'A' < 'B' alphabetically
        self.assertTrue(A < B)
        self.assertFalse(B < A)

if __name__ == '__main__':
    unittest.main()
