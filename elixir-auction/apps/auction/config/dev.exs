import Config

config :auction, Auction.Repo,
  loggers: [{Ecto.LogEntry, :log, [:debug]}]
