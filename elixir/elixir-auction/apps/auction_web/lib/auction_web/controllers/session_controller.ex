defmodule AuctionWeb.SessionController do
  use AuctionWeb, :controller

  def new(conn, _params) do
    user = Auction.new_user()
    render(conn, :new, changeset: user)
  end

  def create(conn, %{"user" => %{"username" => username, "password" => password}}) do
    case Auction.get_user_by_username_and_password(username, password) do
      %Auction.User{} = user ->
        conn
        |> put_session(:user_id, user.id)
        |> put_flash(:info, "Successfully logged in")
        |> redirect(to: ~p"/items")
      _ ->
        changeset = Auction.User.changeset(%Auction.User{}, %{})
        conn
        |> put_flash(:error, "That username and password combination cannot be found")
        |> render("new.html", changeset: changeset)
    end
  end

  @spec delete(Plug.Conn.t(), any()) :: Plug.Conn.t()
  def delete(conn, _params) do
    conn
    |> clear_session()
    |> configure_session(drop: true)
    |> redirect(to: ~p"/")
  end
end
