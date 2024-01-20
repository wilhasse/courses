import Config

config :music_db, Auction.Repo,
  pool: Ecto.Adapters.SQL.Sandbox,
  adapter: Ecto.Adapters.MyXQL,
  username: "root",
  password: "07farm",
  database: "auction",
  hostname: "localhost"
