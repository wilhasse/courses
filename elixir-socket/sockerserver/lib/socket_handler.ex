defmodule SocketServer.SocketHandler do
  @behaviour :cowboy_websocket

  @spec init(atom() | %{:path => any(), optional(any()) => any()}, any()) ::
          {:cowboy_websocket, atom() | %{:path => any(), optional(any()) => any()},
           %{registry_key: any()}}
  def init(request, _state) do
    state = %{registry_key: request.path}

    {:cowboy_websocket, request, state}
  end

  def websocket_init(state) do
    Registry.SocketServer
    |> Registry.register(state.registry_key, {})

    {:ok, state}
  end

  def websocket_handle({:text, json}, state) do
    payload = Jason.decode!(json)
    message = payload["data"]["message"]

    Registry.SocketServer
    |> Registry.dispatch(state.registry_key, fn(entries) ->
      for {pid, _} <- entries do
        if pid != self() do
          Process.send(pid, message, [])
        end
      end
    end)

    {:reply, {:text, message}, state}
  end

  def websocket_info(info, state) do
    {:reply, {:text, info}, state}
  end
end
