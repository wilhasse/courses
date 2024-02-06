defmodule HttpServer.PageController do

  @templates_path Path.expand("../templates", __DIR__)

  defmodule Bear do
    defstruct id: nil, name: "", type: "", hibernating: false
  end

  def list_bears do
    [
      %Bear{id: 1, name: "Teddy", type: "Brown", hibernating: true},
      %Bear{id: 2, name: "Smokey", type: "Black"},
      %Bear{id: 3, name: "Bob", type: "Brown"},
    ]
  end

  def index(conv) do
    bears = list_bears()

    render(conv, "index.eex", bears: bears)
  end

  def create(conv, %{"name" => name, "type" => type}) do
    %{ conv | status: 201,
              resp_body: "Created a #{type} bear named #{name}!" }
  end

  defp render(conv, template, bindings) do
    content =
      @templates_path
      |> Path.join(template)
      |> EEx.eval_file(bindings)

    %{ conv | status: 200, resp_body: content }
  end

end
