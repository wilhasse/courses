defmodule AuctionWeb.PageController do
  use AuctionWeb, :controller

  def index(conn, _params) do
    # The home page is often custom made,
    # so skip the default app layout.
    render(conn, :home)
  end
end
