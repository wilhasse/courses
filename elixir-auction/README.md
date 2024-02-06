# ElixirAuction


mix deps.get

mix phx.routes AuctionWeb.Router

mix ecto.create

mix ecto.migrations

mix ecto.migrate

mix ecto.migrations

iex -S mix phx.server

> Auction.list_items()

> Auction.insert_item(%{title: "Meu carro", description: "Carro", ends_at: DateTime.from_naive!(~N[2024-01-29 12:00:00], "Etc/UTC")})
