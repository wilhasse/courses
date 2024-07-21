defmodule TcpSocketServer do
  require Logger

  def start(port) do
    spawn fn ->
      case :gen_tcp.listen(port, [:binary, active: false, reuseaddr: true]) do
        {:ok, socket} ->
          Logger.info("Receiver listening on port #{port}")
          accept_connection(socket)
        {:error, reason} ->
          Logger.error("Could not start Receiver: #{inspect reason}.")
      end
    end
  end

  def accept_connection(socket) do
    case :gen_tcp.accept(socket) do
      {:ok, client} ->
        spawn fn ->
          {:ok, buffer_pid} = Buffer.create()
          Process.flag(:trap_exit, true)
          serve(client, buffer_pid)
        end
        loop_accept(socket)
      {:error, :closed} ->
        Logger.warning("#{__MODULE__} restarted, so the listen socket closed.")
      {:error, reason} ->
        Logger.error("ACCEPT ERROR: #{inspect reason}")
    end
  end

  def loop_accept(socket) do
    case :gen_tcp.accept(socket) do
      {:ok, client} ->
        spawn fn ->
          {:ok, buffer_pid} = Buffer.create()
          Process.flag(:trap_exit, true)
          serve(client, buffer_pid)
        end
        loop_accept(socket)
      {:error, :closed} ->
        Logger.warning("#{__MODULE__} socket closed, stopping accept loop.")
      {:error, reason} ->
        Logger.error("Failed to accept connection: #{inspect reason}")
        :timer.sleep(1000)
        loop_accept(socket)
    end
  end

  def serve(socket, buffer_pid) do
    case socket |> :gen_tcp.recv(0) do
      {:ok, data} ->
        buffer_pid = maybe_recreate_buffer(buffer_pid)
        Buffer.receive(buffer_pid, data)
        serve(socket, buffer_pid)
      {:error, reason} ->
        Logger.info("Socket terminating: #{inspect reason}")
    end
  end

  defp maybe_recreate_buffer(original_pid) do
    receive do
      {:EXIT, ^original_pid, _reason} ->
        {:ok, new_buffer_pid} = Buffer.create()
        new_buffer_pid
    after
      10 ->
        original_pid
    end
  end
end
