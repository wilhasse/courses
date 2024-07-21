# PhoenixLive

To start your Phoenix server:

- Run `mix setup` to install and setup dependencies
- Start Phoenix endpoint with `mix phx.server` or inside IEx with `iex -S mix phx.server`

Now you can visit [`localhost:4000`](http://localhost:4000) from your browser.

# Usefull commands

## install phoenix

mix archive.install hex phx_new

## create a new phoenix project without Ecto (inside umbrella)

mix phx.new.web auction_web --no-ecto

## list routes available

mix phx.routes

## run phoenix server

mix phx.server

## run phoenix server inside interactive shell

iex -S mix phx.server

## erlang observer

In mix.exs application add extra_applications: [ … , :observer, :wx, :runtime_tools]
iex -S mix
iex(1)> :observer.start

Phoenix Channel

## create phoenix application

mix phx.new phoenix_channel --no-ecto

## create a channel

mix phx.gen.channel Log

## change config\dev.ex (allow external connections)

http: [ip: {0, 0, 0, 0}, port: 4000],

## assets\js\user_socket.js change

let channel = socket.channel("log:42", {})

## run

iex -S mix phx.server
