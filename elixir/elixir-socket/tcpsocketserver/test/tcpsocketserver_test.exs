defmodule TcpsocketserverTest do
  use ExUnit.Case
  doctest Tcpsocketserver

  test "greets the world" do
    assert Tcpsocketserver.hello() == :world
  end
end
