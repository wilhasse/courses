defmodule SocketServer.MixProject do
  use Mix.Project

  def project do
    [
      app: :socketserver,
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      mod: {SocketServer, []},
      extra_applications: [:logger, :observer, :wx, :runtime_tools]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:cowboy, "~> 2.10"},
      {:plug, "~> 1.15.3"},
      {:plug_cowboy, "~> 2.7.0"},
      {:jason, "~> 1.4.1"}
    ]
  end
end
