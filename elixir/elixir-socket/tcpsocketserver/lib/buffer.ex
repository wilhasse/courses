defmodule Buffer do
  @moduledoc "Buffer the data received to parse out statements"

  use GenServer
  require Logger

  @eol <<10>>
  @initial_state ""

  defmodule MessageSink do
    def receive(message, time) do
      time_string = Timex.format!(time, "%Y-%m-%d %H:%M:%S", :strftime)
      IO.puts("#{time_string} #{message}")
    end
  end

  def create do
    GenServer.start_link(__MODULE__, @initial_state)
  end

  def init(init_arg) do
    {:ok, init_arg}
  end

  def receive(pid \\ __MODULE__, data) do
    GenServer.cast(pid, {:receive, data})
  end

  def handle_cast({:receive, data}, buffer) do
    buffer
    |> append(data)
    |> process
  end

  defp append(buffer, ""), do: buffer

  defp append(buffer, data), do: buffer <> data

  defp process(buffer) do
    case extract(buffer) do
      {:statement, buffer, statement} ->
        MessageSink.receive(statement, Timex.now())
        process(buffer)
      {:nothing, buffer} ->
        {:noreply, buffer}
    end
  end

  defp extract(buffer) do
    case String.split(buffer, @eol, parts: 2) do
      [match, rest] -> {:statement, rest, match}
      [rest] -> {:nothing, rest}
    end
  end
end
