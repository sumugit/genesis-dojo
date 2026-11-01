"""Simulator backends. Only `make_sim` should be imported by task code."""

from .interface import QuadrupedSim, SimState


def make_sim(name: str, **kwargs) -> QuadrupedSim:
    if name == "genesis":
        from .genesis_backend import GenesisGo2Sim

        return GenesisGo2Sim(**kwargs)
    raise ValueError(f"unknown simulator backend: {name}")


__all__ = ["QuadrupedSim", "SimState", "make_sim"]
