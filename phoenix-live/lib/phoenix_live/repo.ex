defmodule PhoenixLive.Repo do
  use Ecto.Repo,
    otp_app: :phoenix_live,
    adapter: Ecto.Adapters.Postgres
end
