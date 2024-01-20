import Config

# We don't run a server during test. If one is required,
# you can enable the server option below.
config :auction_web, AuctionWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4002],
  secret_key_base: "WQ6FTibk+5StmVuw6EWlY6Inb7PHH3lXWK+9HqV/KIaDqCwr4UnCFcto4MzMgsEk",
  server: false
