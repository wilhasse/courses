defmodule Websockserver.MixProject do
  use Mix.Project

  def project do
    [
      app: :websockserver,
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger,:observer, :wx, :runtime_tools],
      mod: {WebSockServer, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:plug_cowboy, "~> 2.7.0"},
      {:websock_adapter, "~> 0.5.5"},
    ]
  end
end
