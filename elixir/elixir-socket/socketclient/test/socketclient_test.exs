defmodule SocketclientTest do
  use ExUnit.Case
  doctest Socketclient

  test "greets the world" do
    assert Socketclient.hello() == :world
  end
end
