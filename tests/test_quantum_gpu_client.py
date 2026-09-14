from typing import Any

import pytest

from thrml.quantum_gpu_client import GPUCompletionEvent, IHardwareQPUInterface, QPUEvent, QuantumGPUClient


class MockHardwareQPUInterface(IHardwareQPUInterface):
    def __init__(self):
        self.dma_allocations = []
        self.qpu_events = []
        self.gpu_events = []

    def setup_dma(self, buffer_size: int, dtype: Any) -> Any:
        handle = f"dma_handle_{len(self.dma_allocations)}"
        self.dma_allocations.append((buffer_size, dtype, handle))
        return handle

    def register_qpu_event(self, signal_address: int) -> Any:
        handle = f"qpu_event_handle_{signal_address}"
        self.qpu_events.append((signal_address, handle))
        return handle

    def register_gpu_completion_event(self, signal_address: int) -> Any:
        handle = f"gpu_event_handle_{signal_address}"
        self.gpu_events.append((signal_address, handle))
        return handle


def test_quantum_gpu_client_allocation():
    interface = MockHardwareQPUInterface()
    client = QuantumGPUClient(interface)

    size = 1024
    dtype = "float32"
    handle = client.allocate_qpu_gpu_buffer(size, dtype)

    assert handle == "dma_handle_0"
    assert len(interface.dma_allocations) == 1
    assert interface.dma_allocations[0] == (size, dtype, handle)


def test_quantum_gpu_client_events():
    interface = MockHardwareQPUInterface()
    client = QuantumGPUClient(interface)

    qpu_addr = 0x1000
    qpu_event = client.create_qpu_event(qpu_addr)

    assert isinstance(qpu_event, QPUEvent)
    assert qpu_event.signal_address == qpu_addr
    assert qpu_event.handle == f"qpu_event_handle_{qpu_addr}"
    assert len(interface.qpu_events) == 1

    gpu_addr = 0x2000
    gpu_event = client.create_gpu_completion_event(gpu_addr)

    assert isinstance(gpu_event, GPUCompletionEvent)
    assert gpu_event.signal_address == gpu_addr
    assert gpu_event.handle == f"gpu_event_handle_{gpu_addr}"
    assert len(interface.gpu_events) == 1


def test_workflow_simulation():
    interface = MockHardwareQPUInterface()
    client = QuantumGPUClient(interface)

    # 1. Initialization
    buffer_handle = client.allocate_qpu_gpu_buffer(1024, "complex128")

    qpu_signal_addr = 0xCAFE
    gpu_signal_addr = 0xBABE

    event_from_qpu = client.create_qpu_event(qpu_signal_addr)
    event_to_qpu = client.create_gpu_completion_event(gpu_signal_addr)

    # 2. Critical Path Simulation (Logical check)
    # GPU prepares state (mock)
    # GPU signals QPU
    event_to_qpu.record()  # Should be non-blocking/successful

    # QPU execution (simulated by us just proceeding)

    # QPU writes results and signals GPU (simulated)
    # GPU waits for QPU
    event_from_qpu.wait()  # Should be non-blocking/successful

    # Verify handles were created correctly
    assert buffer_handle is not None
    assert event_from_qpu.signal_address == qpu_signal_addr
    assert event_to_qpu.signal_address == gpu_signal_addr


if __name__ == "__main__":
    pytest.main([__file__])
