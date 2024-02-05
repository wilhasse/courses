defmodule SocketClient do
  use WebSockex
  require Logger

  # Run Test
  # {:ok, pid} = WebSockex.start_link(url, __MODULE__, args, connect_opts)
  # WebSockex.send_frame(pid, {:text, "\{     \"data\": \{         \"message\": \"1000\"     \} \}"})

  @echo_server "ws://localhost:4000/ws/chat"
  def start_link(opts \\ []) do
    WebSockex.start_link(@echo_server, __MODULE__, %{}, opts)
  end

  def handle_frame({:text, "please reply" = msg}, state) do
    Logger.info("Echo server says, #{msg}")
    reply = "Back, atcha!"

    Logger.info("Sent to Echo server: #{reply}")
    {:reply, {:text, reply}, state}
  end

  def handle_frame({:text, "shut down"}, state) do
    Logger.info("shutting down...")
    {:close, state}
  end

  def handle_frame({:text, msg}, state) do
    Logger.info("Echo server says, #{msg}")
    {:ok, state}
  end

  def handle_disconnect(%{reason: reason}, state) do
    Logger.info("Disconnect with reason: #{inspect reason}")
    {:ok, state}
  end
end
