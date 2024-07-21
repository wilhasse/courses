defmodule WebsockserverTest do
  use ExUnit.Case
  doctest Websockserver

  test "greets the world" do
    assert Websockserver.hello() == :world
  end
end
