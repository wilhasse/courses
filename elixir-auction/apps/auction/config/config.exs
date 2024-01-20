import Config

config :auction, ecto_repos: [Auction.Repo]

config :auction, Auction.Repo,
  database: "auction",
  username: "root",
  password: "07farm",
  hostname: "localhost"

# Import environment specific config. This must remain at the bottom
# of this file so it overrides the configuration defined above.
import_config "#{config_env()}.exs"
