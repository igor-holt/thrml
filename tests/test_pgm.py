import jax
import jax.numpy as jnp
from thrml.pgm import AbstractNode

def test_abstract_node_ordering():
    def create_node_class(name):
        class Node(AbstractNode):
            pass
        Node.__name__ = name
        Node.__qualname__ = name
        return Node

    A = create_node_class("A")
    B = create_node_class("A") # same name and module to trigger equality fallback

    assert A.__module__ == B.__module__
    assert A.__qualname__ == B.__qualname__

    # Due to _class_id, A should be less than B if A was created first
    assert A < B
    assert not B < A

    sorted_nodes = sorted([B, A])
    assert sorted_nodes == [A, B]

def test_jax_tree_sorting():
    def create_node_class(name):
        class Node(AbstractNode):
            pass
        Node.__name__ = name
        Node.__qualname__ = name
        return Node

    A = create_node_class("Node")
    B = create_node_class("Node")

    # JAX sorts dictionary keys when flattening
    # If keys are unorderable or ordering is inconsistent, this could fail
    d = {B: jnp.array([2.0]), A: jnp.array([1.0])}

    flat, treedef = jax.tree.flatten(d)

    # Expect A to come first in the flattened array, since A < B
    # A corresponds to [1.0], B corresponds to [2.0]
    assert flat[0].item() == 1.0
    assert flat[1].item() == 2.0
