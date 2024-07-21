# Usefull commands

## create umbrella project

> mix new --umbrella auction_umbrella

## create new supervisor project

> mix new auction --sup

## create new project

> mix new example

## compile

> mix compile

## run shell

> iex -S mix
> Example.hello()

## run mix

> mix run -e "Example.hello()"

## search for packages

> mix hex.search ecto

## pulling in dependencies in mix.exs

> mix deps.get

## test auction

```bash
iex -S mix

query>
Auction.list_items()

insert>
{:ok, item} =
Auction.insert_item(
%{title: "WarGames Bluray",
description: "Computer games and thermonuclear war",
ends_at: DateTime.from_naive!(~N[2019-02-22 11:43:39], "Etc/UTC")}
)
```

# repo.ex in lib\project

```bash
defmodule MusicDB.Repo do
use Ecto.Repo,
otp_app: :music_db,
adapter: Ecto.Adapters.MyXQL
end
```

# config/config.exs

```bash
config :auction, ecto_repos: [Auction.Repo]

config :auction, Auction.Repo,
database: "auction",
username: "root",
password: "",
hostname: "localhost"
```

# create database

> mix ecto.create

# create migration:

> mix ecto.gen.migration create_items

# add table definition in priv/repo/migrations

```bash
def change do
create table("items") do
add :title, :string
timestamps()
end
end
```

# create schema migration

> mix ecto.migrations

# run migration and create table

> mix ecto.migrate
