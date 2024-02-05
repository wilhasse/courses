defmodule SocketServer.Plugins do

  def log(text) do
    if Mix.env == :dev do
      IO.puts "Plugins: #{text}"
    end
    text
  end
end
