"""Triton kernels behind registry ops. Imported by an op's implementation
factory, never at declaration, so declaring an op does not import Triton."""
