from thrml.conditional_samplers import AbstractConditionalSampler
import jax.numpy as jnp

class DummySampler(AbstractConditionalSampler):
    def init(self):
        return None
    def sample(self, key, interactions, interaction_active, interaction_states, sampler_state, sd):
        return jnp.zeros(()), None
