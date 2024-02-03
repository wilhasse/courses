defmodule SockerserverTest do
  use ExUnit.Case
  doctest Sockerserver

  test "greets the world" do
    assert Sockerserver.hello() == :world
  end
end
