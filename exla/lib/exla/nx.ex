defmodule EXLA.Nx do
  @moduledoc """
  EXLA-specific Nx operations for distributed computing.

  These operations are only available when using EXLA as the backend
  and provide access to collective communication primitives.

  ## All-Reduce

  The `all_reduce/3` operation sums (or reduces with another op) tensor
  values across all replicas in an SPMD computation.

  ## Example

      defmodule MyModel do
        import Nx.Defn

        defn forward(x, opts \\\\ []) do
          partial = some_computation(x)

          # Sum across all 4 GPUs
          EXLA.Nx.all_reduce(partial, :sum, replica_groups: [[0, 1, 2, 3]])
        end
      end

  ## SPMD Execution

  For the all-reduce to actually communicate between devices, the computation
  must be compiled and executed in SPMD mode with `num_replicas > 1`.
  """

  import Nx.Defn

  @doc """
  All-reduce operation that sums tensor values across replicas.

  This is a collective operation that synchronizes values across all
  devices in the specified replica groups.

  ## Arguments

    * `tensor` - The tensor to reduce
    * `op` - Reduction operation: `:sum`, `:max`, `:min`, or `:product`
    * `opts` - Options:
      * `:replica_groups` - List of replica group lists (required)
        Example: `[[0, 1, 2, 3]]` for 4-way all-reduce

  ## Returns

  A tensor with the same shape as the input, containing the reduced
  values. All replicas in the same group will have identical output.

  ## Example

      # 4-GPU tensor parallelism - sum partial results
      result = EXLA.Nx.all_reduce(partial, :sum, replica_groups: [[0, 1, 2, 3]])

      # 2x2 TP/DP configuration
      # TP groups: [0,1] and [2,3], DP groups: [0,2] and [1,3]
      result = EXLA.Nx.all_reduce(partial, :sum, replica_groups: [[0, 1], [2, 3]])
  """
  deftransform all_reduce(tensor, op \\ :sum, opts \\ []) when op in [:sum, :max, :min, :product] do
    replica_groups = Keyword.get(opts, :replica_groups, [[0, 1]])

    # Create the all_reduce expression using Nx.Defn.Expr.optional
    # This will be intercepted by EXLA.Defn and lowered to MLIR all_reduce
    Nx.Defn.Expr.optional(
      :all_reduce,
      [tensor, [op: op, replica_groups: replica_groups]],
      fn tensor, _opts -> tensor end
    )
  end
end
