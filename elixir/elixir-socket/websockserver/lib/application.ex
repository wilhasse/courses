defmodule WebSockServer do
  use Application

  def start(_type, _args) do
    children = [
      {Plug.Cowboy, scheme: :http, plug: MyPlug, options: [port: 4000]}
    ]

    opts = [strategy: :one_for_one, name: WebSockServer.Supervisor]
    IO.puts  "Application started ..."
    Supervisor.start_link(children, opts)
  end
end
