import jax
import jax.numpy as jnp
import time
import numpy as np
import cProfile
import pstats
import io

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.interaction import InteractionGroup
from thrml.pgm import CategoricalNode
from dummy_sampler import DummySampler
from thrml.block_sampling import BlockSamplingProgram

def bench():
    # Construct a big dummy PGM
    n_blocks = 50
    n_nodes_per_block = 200
    n_interactions = 500
    n_states = 2

    blocks = []
    for i in range(n_blocks):
        nodes = [CategoricalNode() for j in range(n_nodes_per_block)]
        blocks.append(Block(nodes))

    gibbs_spec = BlockGibbsSpec(blocks, [])

    interaction_groups = []
    for i in range(n_interactions):
        head_block = blocks[i % n_blocks]
        tail_block = blocks[(i + 1) % n_blocks]

        interaction = jnp.zeros((n_nodes_per_block, n_nodes_per_block, n_states, n_states))
        ig = InteractionGroup(interaction, head_block, [tail_block])
        interaction_groups.append(ig)

    samplers = [DummySampler() for _ in range(n_blocks)]

    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    program = BlockSamplingProgram(gibbs_spec, samplers, interaction_groups)
    end_time = time.time()

    pr.disable()
    s = io.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20)
    print(s.getvalue())

    print(f"Time to create BlockSamplingProgram: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    bench()
