defmodule HttpserverTest do
  use ExUnit.Case
  doctest Httpserver

  test "greets the world" do
    assert Httpserver.hello() == :world
  end
end
