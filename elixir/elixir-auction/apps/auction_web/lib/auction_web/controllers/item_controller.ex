defmodule AuctionWeb.ItemController do
  use AuctionWeb, :controller

  def index(conn, _) do
    items = Auction.list_items()
    render(conn, :item, items: items)
  end

  def show(conn, %{"id" => id}) do
    item = Auction.get_item_with_bids(id)
    bid = Auction.new_bid()
    render(conn, "show.html", item: item, bid: bid)
  end

  def new(conn, _params) do
    changeset = Auction.new_item
    render(conn, :new, changeset: changeset)
  end

  @spec create(Plug.Conn.t(), map()) :: Plug.Conn.t()
  def create(conn, %{"item" => item_params}) do
    case Auction.insert_item(item_params) do
      {:ok, item} -> redirect(conn, to: Routes.item_path(conn, :show, item))
      {:error, item} -> render(conn, "new.html", item: item)
    end
  end

  def edit(conn, %{"id" => id}) do
    item = Auction.edit_item(id)
    render(conn, "edit.html", item: item)
  end

  def update(conn, %{"id" => id, "item" => item_params}) do
    item = Auction.get_item(id)
    case Auction.update_item(item, item_params) do
      {:ok, item} -> redirect(conn, to: Routes.item_path(conn, :show, item))
      {:error, item} -> render(conn, "edit.html", item: item)
    end
  end
end
