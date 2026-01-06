defmodule EXLA.SPMD do
  @moduledoc """
  Single-Program Multiple-Data (SPMD) execution for multi-GPU tensor parallelism.

  This module provides utilities for running computations across multiple GPUs
  where each GPU executes the same program on different data (or the same data
  with different roles in collective operations like all-reduce).

  ## Example: Simple SPMD

      # Build a simple SPMD function
      typespec = EXLA.Typespec.tensor({:f, 32}, {4})

      spmd = EXLA.SPMD.build([typespec, typespec], [typespec], fn builder ->
        [a, b] = EXLA.MLIR.Function.get_arguments(builder)
        result = EXLA.MLIR.Value.add(a, b, typespec)
        [result]
      end, num_replicas: 2)

      # Run on 2 GPUs
      results = EXLA.SPMD.run(spmd, [
        [tensor_a, tensor_b],  # GPU 0 inputs
        [tensor_a, tensor_b]   # GPU 1 inputs
      ])

  ## Example: All-Reduce

      # Function with all-reduce for tensor parallelism
      spmd = EXLA.SPMD.build([typespec], [typespec], fn builder ->
        [x] = EXLA.MLIR.Function.get_arguments(builder)
        # Each GPU computes partial result, all-reduce sums them
        reduced = EXLA.MLIR.Value.all_reduce(x, :sum, [[0, 1]], typespec)
        [reduced]
      end, num_replicas: 2)

      # GPU 0 has [1,2,3,4], GPU 1 has [5,6,7,8]
      # After all-reduce, both get [6,8,10,12]
      results = EXLA.SPMD.run(spmd, [
        [Nx.tensor([1.0, 2.0, 3.0, 4.0])],
        [Nx.tensor([5.0, 6.0, 7.0, 8.0])]
      ])
  """

  alias EXLA.{BinaryBuffer, DeviceBuffer, Executable, Client}
  alias EXLA.MLIR.{ContextPool, Module, Function, Value}

  defstruct [:executable, :num_replicas, :input_typespecs, :output_typespecs]

  @doc """
  Builds an SPMD executable from an MLIR builder function.

  This provides direct access to MLIR for building SPMD programs with
  collective operations like all-reduce.

  ## Arguments

    * `input_typespecs` - List of EXLA.Typespec for inputs
    * `output_typespecs` - List of EXLA.Typespec for outputs
    * `builder_fun` - Function that receives MLIR builder and returns output values
    * `opts` - Options including:
      * `:num_replicas` - Number of GPUs (required)
      * `:client` - EXLA client name (default: `:cuda`)

  ## Returns

    An `EXLA.SPMD` struct for use with `run/2`.
  """
  def build(input_typespecs, output_typespecs, builder_fun, opts) when is_function(builder_fun, 1) do
    num_replicas = Keyword.fetch!(opts, :num_replicas)
    client_name = Keyword.get(opts, :client, :cuda)
    client = Client.fetch!(client_name)

    # Build MLIR module - we need to capture the module from within the builder
    # MLIR.Module.new returns the builder's output, but we need the module for compilation
    executable = ContextPool.checkout(fn context ->
      module_ref = EXLA.NIF.mlir_new_module(context)
      module = %Module{ref: module_ref}

      # Create the main function
      function = create_function(module, "main", input_typespecs, output_typespecs, true)

      # Call the builder to populate the function
      outputs = builder_fun.(function)

      # Add return statement
      Value.return(function, outputs)

      # Compile with num_replicas for SPMD execution
      Module.compile(
        module,
        client,
        input_typespecs,
        output_typespecs,
        num_replicas: num_replicas
      )
    end)

    %__MODULE__{
      executable: executable,
      num_replicas: num_replicas,
      input_typespecs: input_typespecs,
      output_typespecs: output_typespecs
    }
  end

  # Create a function in the module (adapted from MLIR.Module)
  defp create_function(%Module{ref: module_ref} = module, name, arg_typespecs, return_typespecs, is_public) do
    arg_types = Value.typespecs_to_mlir_types(arg_typespecs)
    return_types = Value.typespecs_to_mlir_types(return_typespecs)

    ref = EXLA.NIF.mlir_create_function(
      module_ref,
      name,
      arg_types,
      return_types,
      is_public
    )

    %Function{module: module, ref: ref, name: name, return_typespecs: return_typespecs}
  end

  @doc """
  Runs an SPMD program with replica-batched inputs.

  ## Arguments

    * `spmd` - The SPMD struct from `build/4`
    * `replica_inputs` - List of input lists, one per replica.
      Length must equal `num_replicas`. Each inner list must match `input_typespecs`.

  ## Returns

    List of output lists, one per replica.

  ## Example

      # 2-replica SPMD with 2 inputs each
      results = EXLA.SPMD.run(spmd, [
        [tensor_a_gpu0, tensor_b_gpu0],
        [tensor_a_gpu1, tensor_b_gpu1]
      ])
      # => [[output_gpu0], [output_gpu1]]
  """
  def run(%__MODULE__{} = spmd, replica_inputs) when is_list(replica_inputs) do
    %{
      executable: executable,
      num_replicas: num_replicas,
      input_typespecs: input_typespecs,
      output_typespecs: output_typespecs
    } = spmd

    if length(replica_inputs) != num_replicas do
      raise ArgumentError,
            "Expected #{num_replicas} replica inputs, got #{length(replica_inputs)}"
    end

    # Convert each replica's inputs to buffers
    buffer_inputs =
      Enum.map(replica_inputs, fn inputs ->
        if length(inputs) != length(input_typespecs) do
          raise ArgumentError,
                "Expected #{length(input_typespecs)} inputs per replica, got #{length(inputs)}"
        end

        Enum.zip_with(inputs, input_typespecs, fn input, typespec ->
          to_buffer(input, typespec)
        end)
      end)

    # Run executable with all replicas' inputs
    results = Executable.run(executable, buffer_inputs)

    # Convert outputs back to tensors
    Enum.map(results, fn replica_outputs ->
      Enum.zip_with(replica_outputs, output_typespecs, fn output, typespec ->
        buffer_to_tensor(output, typespec)
      end)
    end)
  end

  @doc """
  Creates an SPMD program that performs all-reduce across replicas.

  This is a convenience function for the common tensor parallelism pattern
  where partial results are summed across GPUs.

  ## Example

      # Create all-reduce for 2 GPUs
      spmd = EXLA.SPMD.all_reduce({:f, 32}, {1024}, num_replicas: 2)

      # Each GPU provides its partial result
      [[sum], [sum]] = EXLA.SPMD.run(spmd, [
        [partial_gpu0],
        [partial_gpu1]
      ])
      # sum = partial_gpu0 + partial_gpu1 (same on both GPUs)
  """
  def all_reduce(type, shape, opts) do
    num_replicas = Keyword.fetch!(opts, :num_replicas)
    reduction_op = Keyword.get(opts, :op, :sum)

    typespec = EXLA.Typespec.tensor(type, shape)
    replica_groups = [Enum.to_list(0..(num_replicas - 1))]

    build([typespec], [typespec], fn builder ->
      [input] = Function.get_arguments(builder)
      result = Value.all_reduce(input, reduction_op, replica_groups, typespec)
      [result]
    end, opts)
  end

  # Convert Nx tensor to EXLA buffer
  defp to_buffer(%Nx.Tensor{} = tensor, typespec) do
    %BinaryBuffer{data: Nx.to_binary(tensor), typespec: typespec}
  end

  defp to_buffer(%DeviceBuffer{} = buf, _typespec), do: buf
  defp to_buffer(%BinaryBuffer{} = buf, _typespec), do: buf

  # Convert EXLA buffer back to Nx tensor
  defp buffer_to_tensor(%DeviceBuffer{typespec: typespec} = buf, _typespec) do
    data = DeviceBuffer.read(buf)
    Nx.from_binary(data, typespec.type) |> Nx.reshape(typespec.shape)
  end

  defp buffer_to_tensor(%BinaryBuffer{data: data, typespec: typespec}, _typespec) do
    Nx.from_binary(data, typespec.type) |> Nx.reshape(typespec.shape)
  end
end
