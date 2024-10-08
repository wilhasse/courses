defmodule SocketServer do
  use Application

  def start(_type, _args) do
    children = [
      Plug.Cowboy.child_spec(
        scheme: :http,
        plug: SocketServer.Router,
        options: [
          dispatch: dispatch(),
          port: 4000
        ]
      ),
      Registry.child_spec(
        keys: :duplicate,
        name: Registry.SocketServer

      )
    ]

    opts = [strategy: :one_for_one, name: SocketServer.Application]
    Supervisor.start_link(children, opts)
  end

  defp dispatch do
    [
      {:_,
        [
          {"/ws/[...]", SocketServer.SocketHandler, []},
          {:_, Plug.Cowboy.Handler, {SocketServer.Router, []}}
        ]
      }
    ]
  end
end
