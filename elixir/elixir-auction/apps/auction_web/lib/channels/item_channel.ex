defmodule AuctionWeb.ItemChannel do
  use AuctionWeb, :channel

  @impl true
  def join("item:" <> _item_id, _params, socket) do
    {:ok, socket}
  end

  # Channels can be used in a request/response fashion
  # by sending replies to requests from the client
  @impl true
  def handle_in("ping", payload, socket) do
    {:reply, {:ok, payload}, socket}
  end

  @impl true
  def handle_in("new_bid", params, socket) do
    broadcast!(socket, "new_bid", params)
    {:noreply, socket}
  end
end
