defmodule HttpServer.Handler do

  @moduledoc "Handles HTTP requests."

  alias HttpServer.Conv
  alias HttpServer.PageController

  @pages_path Path.expand("../pages", __DIR__)

  import HttpServer.Parser, only: [parse: 1]

  @doc "Transforms the request into a response."
  def handle(request) do
    request
    |> parse
    |> route
    |> format_response
  end

  def route(%Conv{ method: "GET", path: "/test" } = conv) do
    IO.puts("GET test")
    %{ conv | status: 200, resp_body: "Foo, Boo" }
  end

  def route(%Conv{method: "POST", path: "/test"} = conv) do
    IO.puts("POST test")
    PageController.create(conv, conv.params)
  end

  def route(%Conv{ method: "GET", path: "/list" } = conv) do
    PageController.index(conv)
  end

  def route(%Conv{method: "GET", path: "/about"} = conv) do
      IO.puts("about")
      @pages_path
      |> Path.join("about.html")
      |> File.read
      |> handle_file(conv)
  end

  def route(%Conv{ path: path } = conv) do
    %{ conv | status: 404, resp_body: "No #{path} here!"}
  end

  def handle_file({:ok, content}, conv) do
    %{ conv | status: 200, resp_body: content }
  end

  def handle_file({:error, :enoent}, conv) do
    %{ conv | status: 404, resp_body: "File not found!" }
  end

  def handle_file({:error, reason}, conv) do
    %{ conv | status: 500, resp_body: "File error: #{reason}" }
  end

  def format_response(%Conv{} = conv) do
    """
    HTTP/1.1 #{Conv.full_status(conv)}\r
    Content-Type: #{conv.resp_content_type}\r
    Content-Length: #{String.length(conv.resp_body)}\r
    \r
    #{conv.resp_body}
    """
  end

end
