import jax
import jax.numpy as jnp
import unittest
from thrml import block_management
from thrml.pgm import AbstractNode

class ScalarNode(AbstractNode):
    pass

class TestBlockScalar(unittest.TestCase):
    def test_scalar_state_shapes(self):
        """
        Verify that single-block scalar states are correctly expanded to (1,)
        to match the (N,) shape of multi-block states.
        """
        node_shape_dtypes = {
            ScalarNode: jax.ShapeDtypeStruct((), jnp.float32)
        }

        # Case 1: Single block
        blocks_1 = [block_management.Block([ScalarNode()])]
        spec_1 = block_management.BlockSpec(blocks_1, node_shape_dtypes)
        # Scalar state (shape ())
        block_state_1 = [jnp.array(1.0)]

        global_state_1 = block_management.block_state_to_global(block_state_1, spec_1)

        # We expect (1,) not ()
        self.assertEqual(global_state_1[0].shape, (1,), "Single block scalar state should be expanded to (1,)")

        # Case 2: Two blocks
        blocks_2 = [block_management.Block([ScalarNode()]), block_management.Block([ScalarNode()])]
        spec_2 = block_management.BlockSpec(blocks_2, node_shape_dtypes)
        # Scalar states
        block_state_2 = [jnp.array(1.0), jnp.array(2.0)]

        global_state_2 = block_management.block_state_to_global(block_state_2, spec_2)

        # We expect (2,)
        self.assertEqual(global_state_2[0].shape, (2,), "Two block scalar states should be stacked to (2,)")

if __name__ == "__main__":
    unittest.main()
