defmodule EXLA.Collective do
  @moduledoc """
  Collective communication operations for distributed tensor computation.

  These operations synchronize data across multiple devices in a mesh.
  They are essential for tensor parallelism where partial results from
  different GPUs need to be combined.

  ## Overview

  In tensor parallelism, model weights are sharded across GPUs. After certain
  operations (row-parallel linear layers), partial results need to be summed
  across all GPUs. This is done via `all_reduce`.

  ## Example

      # Create a device mesh with 2 GPUs
      # Each GPU computes a partial result
      # Sum partial results across all GPUs using defn

      defmodule MyModel do
        import Nx.Defn

        defn forward(x, weight) do
          # Local computation
          partial = Nx.dot(x, weight)

          # All-reduce to sum partial results across devices
          # This will be converted to stablehlo.all_reduce during compilation
          all_reduce_sum(partial)
        end

        # Custom all-reduce operation using EXLA
        defnp all_reduce_sum(tensor) do
          # The replica_groups [[0, 1]] means devices 0 and 1 form one group
          custom_all_reduce(tensor, :sum, [[0, 1]])
        end
      end

  ## Supported Operations

    * `:sum` - Sum of all values across replicas
    * `:max` - Maximum value across replicas
    * `:min` - Minimum value across replicas
    * `:product` - Product of all values across replicas

  ## Notes

  These operations are only available when using the EXLA backend with
  multi-device compilation (SPMD mode). They require:

    * Multiple GPUs or simulated devices via `XLA_FLAGS`
    * EXLA compiled with NCCL support for GPU collective operations
  """

  alias EXLA.MLIR.Value

  @doc """
  Reduces tensors across all devices using the specified operation.

  The input tensor on each device contains a partial result. After all_reduce,
  each device will have the combined result (e.g., sum of all partial results).

  ## Arguments

    * `tensor` - The input tensor (partial result on each device)
    * `op` - Reduction operation: `:sum`, `:max`, `:min`, `:product`
    * `opts` - Options:
      * `:replica_groups` - List of replica groups (required)
        Each group is a list of device IDs that participate in the reduction.
        For tensor parallelism with 2 GPUs: `[[0, 1]]`
      * `:channel_id` - Optional channel ID for the collective (auto-assigned if nil)
      * `:use_global_device_ids` - Whether to use global device IDs (default: false)

  ## Returns

    The reduced tensor, identical on all devices in each replica group.

  ## Example

      # Each GPU has partial output from row-parallel matmul
      # GPU 0: [[1, 2], [3, 4]]
      # GPU 1: [[5, 6], [7, 8]]

      result = all_reduce(partial, :sum, replica_groups: [[0, 1]])
      # Both GPUs now have: [[6, 8], [10, 12]]

  ## Implementation Note

  This function is designed to be called within an EXLA defn context.
  It creates a `stablehlo.all_reduce` operation in the MLIR graph.
  """
  @spec all_reduce(Value.t(), atom(), keyword()) :: Value.t()
  def all_reduce(%Value{} = tensor, op \\ :sum, opts \\ [])
      when op in [:sum, :max, :min, :product] do
    replica_groups = Keyword.fetch!(opts, :replica_groups)
    channel_id = Keyword.get(opts, :channel_id)
    use_global_device_ids = Keyword.get(opts, :use_global_device_ids, false)

    typespec = Value.get_typespec(tensor)

    Value.all_reduce(tensor, op, replica_groups, typespec,
      channel_id: channel_id,
      use_global_device_ids: use_global_device_ids
    )
  end

  @doc """
  Convenience function for sum reduction across all devices.

  This is the most common all-reduce operation used in tensor parallelism
  to combine partial results from row-parallel linear layers.

  ## Arguments

    * `tensor` - The input tensor to reduce
    * `opts` - Options (see `all_reduce/3`)

  ## Example

      # Sum partial results from row-parallel matmul
      full_result = all_reduce_sum(partial_result, replica_groups: [[0, 1]])
  """
  @spec all_reduce_sum(Value.t(), keyword()) :: Value.t()
  def all_reduce_sum(%Value{} = tensor, opts \\ []) do
    all_reduce(tensor, :sum, opts)
  end

  @doc """
  Convenience function for max reduction across all devices.

  ## Arguments

    * `tensor` - The input tensor to reduce
    * `opts` - Options (see `all_reduce/3`)
  """
  @spec all_reduce_max(Value.t(), keyword()) :: Value.t()
  def all_reduce_max(%Value{} = tensor, opts \\ []) do
    all_reduce(tensor, :max, opts)
  end

  @doc """
  Convenience function for min reduction across all devices.

  ## Arguments

    * `tensor` - The input tensor to reduce
    * `opts` - Options (see `all_reduce/3`)
  """
  @spec all_reduce_min(Value.t(), keyword()) :: Value.t()
  def all_reduce_min(%Value{} = tensor, opts \\ []) do
    all_reduce(tensor, :min, opts)
  end

  @doc """
  Creates replica groups for tensor parallelism.

  Helper function to generate replica groups configuration for
  common tensor parallelism patterns.

  ## Arguments

    * `tp_size` - Number of devices for tensor parallelism
    * `opts` - Options:
      * `:dp_size` - Data parallelism size (default: 1)
        When using both TP and DP, creates separate groups per DP replica.

  ## Returns

    A list of replica groups suitable for `all_reduce/3`.

  ## Examples

      # Simple TP=2 (devices 0,1 form one group)
      replica_groups(2)
      # => [[0, 1]]

      # TP=2 with DP=2 (4 total devices)
      # Devices [0,1] form TP group 0, [2,3] form TP group 1
      replica_groups(2, dp_size: 2)
      # => [[0, 1], [2, 3]]
  """
  @spec replica_groups(pos_integer(), keyword()) :: [[non_neg_integer()]]
  def replica_groups(tp_size, opts \\ []) when tp_size >= 1 do
    dp_size = Keyword.get(opts, :dp_size, 1)

    for dp_rank <- 0..(dp_size - 1) do
      base = dp_rank * tp_size
      Enum.to_list(base..(base + tp_size - 1))
    end
  end
end
