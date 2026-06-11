from torch import nn
import torch

from mstar.distributed.communication import TPCommGroup
from mstar.distributed.utils import divide, split_tensor_along_last_dim


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """

    def __init__(
        self,
        comm_group: TPCommGroup,
        input_size: int,
        output_size: int,
        bias: bool = False,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        # Divide the weight matrix along the last dimension.
        self.tp_rank = comm_group.rank
        self.tp_size = comm_group.world_size
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]

        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, self.input_size_per_partition, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.comm_group = comm_group
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        """Attach the bound ``weight_loader`` to ``self.weight`` (and
        ``self.bias`` if present). Re-run after any ``_apply`` because
        ``to_empty`` / ``.to(...)`` re-allocates Parameters and drops
        attribute attachments on the old objects.
        """
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        return result

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | str | None = None,
    ):
        """Copy this rank's slice of ``loaded_weight`` into ``param``.

        For plain ColumnParallelLinear there's a single shard so
        ``loaded_shard_id`` must be None. Subclasses (``MergedColumnParallelLinear``,
        ``QKVParallelLinear``) override to dispatch by shard id.

        Args:
            param: the destination parameter (``self.weight`` or
                ``self.bias``).
            loaded_weight: the full-checkpoint tensor — shape
                ``(output_size, input_size)`` for weight, ``(output_size,)``
                for bias.
        """
        assert loaded_shard_id is None, (
            f"{type(self).__name__}.weight_loader doesn't take a shard id; "
            f"got {loaded_shard_id!r}"
        )
        # Partition along dim 0 (output dim).
        start = self.tp_rank * self.output_size_per_partition
        shard = loaded_weight.narrow(0, start, self.output_size_per_partition)
        assert param.data.shape == shard.shape, (
            f"weight_loader shape mismatch: param {tuple(param.data.shape)} "
            f"vs shard {tuple(shard.shape)}"
        )
        param.data.copy_(shard)

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, nn.Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        output_parallel = torch.nn.functional.linear(input_, self.weight, bias)

        if self.gather_output and self.tp_size > 1:
            # All-gather across the partitions.
            output = self.comm_group.all_gather(output_parallel, dim=-1)
        else:
            output = output_parallel

        return output


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.
    """

    def __init__(
        self,
        comm_group: TPCommGroup,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        dtype: torch.dtype | None = None,
    ):
        self.output_sizes = output_sizes
        self.tp_rank = comm_group.rank
        self.tp_size = comm_group.world_size

        assert all(output_size % self.tp_size == 0 for output_size in output_sizes)
        super().__init__(
            comm_group=comm_group,
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            dtype=dtype,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ):
        """Copy this rank's slice of one of the merged sub-shards into
        ``param`` at the correct offset.

        ``loaded_weight`` is the full checkpoint tensor for the sub-shard
        identified by ``loaded_shard_id`` (an index into ``output_sizes``).
        """
        assert loaded_shard_id is not None, (
            "MergedColumnParallelLinear.weight_loader requires a loaded_shard_id "
            "(index into output_sizes)."
        )
        assert 0 <= loaded_shard_id < len(self.output_sizes)
        shard_output_size = self.output_sizes[loaded_shard_id]
        shard_per_partition = shard_output_size // self.tp_size

        # Slice this rank's chunk out of the full checkpoint shard.
        src_start = self.tp_rank * shard_per_partition
        src = loaded_weight.narrow(0, src_start, shard_per_partition)

        # Find where this sub-shard sits inside the merged param.
        offset_in_param = sum(
            s // self.tp_size for s in self.output_sizes[:loaded_shard_id]
        )
        dst = param.data.narrow(0, offset_in_param, shard_per_partition)
        assert dst.shape == src.shape, (
            f"weight_loader shape mismatch for shard {loaded_shard_id}: "
            f"dst {tuple(dst.shape)} vs src {tuple(src.shape)}"
        )
        dst.copy_(src)


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.
    """

    def __init__(
        self,
        comm_group: TPCommGroup,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        dtype: torch.dtype | None = None,
        v_head_size: int | None = None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        tp_size = comm_group.world_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (
            self.num_heads * self.head_size
            + self.num_kv_heads * self.head_size
            + self.num_kv_heads * self.v_head_size
        ) * tp_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.v_head_size * tp_size,  # v_proj
        ]

        super().__init__(
            comm_group=comm_group,
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            dtype=dtype,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        """Copy this rank's q / k / v slice into ``param`` at the right
        offset within the merged qkv parameter.

        ``loaded_shard_id`` must be one of ``"q"``, ``"k"``, or ``"v"``.
        For GQA where ``tp_size > total_num_kv_heads`` the K / V heads
        are replicated across ranks (each rank in the same KV-replica
        group loads the same KV head).
        """
        assert loaded_shard_id in ("q", "k", "v"), (
            f"QKVParallelLinear.weight_loader requires loaded_shard_id "
            f"in {{'q','k','v'}}, got {loaded_shard_id!r}"
        )
        if loaded_shard_id == "q":
            per_partition = self.num_heads * self.head_size
            src_start = self.tp_rank * per_partition
            offset_in_param = 0
        elif loaded_shard_id == "k":
            per_partition = self.num_kv_heads * self.head_size
            if self.num_kv_head_replicas > 1:
                kv_head_idx = self.tp_rank // self.num_kv_head_replicas
                src_start = kv_head_idx * per_partition
            else:
                src_start = self.tp_rank * per_partition
            offset_in_param = self.num_heads * self.head_size
        else:  # "v"
            per_partition = self.num_kv_heads * self.v_head_size
            if self.num_kv_head_replicas > 1:
                kv_head_idx = self.tp_rank // self.num_kv_head_replicas
                src_start = kv_head_idx * per_partition
            else:
                src_start = self.tp_rank * per_partition
            offset_in_param = (
                self.num_heads * self.head_size
                + self.num_kv_heads * self.head_size
            )

        src = loaded_weight.narrow(0, src_start, per_partition)
        dst = param.data.narrow(0, offset_in_param, per_partition)
        assert dst.shape == src.shape, (
            f"weight_loader shape mismatch for {loaded_shard_id!r}: "
            f"dst {tuple(dst.shape)} vs src {tuple(src.shape)}"
        )
        dst.copy_(src)


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    """

    def __init__(
        self,
        comm_group: TPCommGroup,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        dtype: torch.dtype | None = None,
        reduce_results: bool = True,
    ):
        super().__init__()
        # Divide the weight matrix along the first dimension.
        self.tp_rank = comm_group.rank
        self.tp_size = comm_group.world_size
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reducing the results, adding bias to the "
                "results can lead to incorrect results"
            )

        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, self.input_size_per_partition, dtype=dtype)
        )
        if bias:
            # Bias is on the output side (not sharded). Each rank holds the
            # full bias; only rank 0 adds it via the forward to avoid
            # double-add under all-reduce.
            self.bias = nn.Parameter(torch.empty(output_size, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.comm_group = comm_group
        self.skip_bias_add = skip_bias_add
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        """Re-attach loaders to ``self.weight`` (sharded) and ``self.bias``
        (replicated). Called from ``__init__`` and re-applied after any
        ``_apply`` transform (``to_empty`` / ``.to(...)``)."""
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self._bias_loader

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        return result

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | str | None = None,
    ):
        """Copy this rank's slice of ``loaded_weight`` into ``param``.

        Partition is along dim 1 (input dim) for the weight.
        """
        assert loaded_shard_id is None, (
            f"RowParallelLinear.weight_loader doesn't take a shard id; "
            f"got {loaded_shard_id!r}"
        )
        start = self.tp_rank * self.input_size_per_partition
        shard = loaded_weight.narrow(1, start, self.input_size_per_partition)
        assert param.data.shape == shard.shape, (
            f"weight_loader shape mismatch: param {tuple(param.data.shape)} "
            f"vs shard {tuple(shard.shape)}"
        )
        param.data.copy_(shard)

    def _bias_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | str | None = None,
    ):
        """Bias is replicated (output dim isn't sharded for row-parallel)."""
        assert loaded_shard_id is None
        param.data.copy_(loaded_weight)


    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, nn.Parameter | None]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        # Matrix multiply.
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = torch.nn.functional.linear(input_parallel, self.weight, bias_)

        if self.reduce_results and self.tp_size > 1:
            output = self.comm_group.all_reduce(output_parallel)
        else:
            output = output_parallel

        return output
