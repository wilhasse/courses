package statefulrouting

import (
    "net/http"
    "sync"

    "github.com/caddyserver/caddy/v2"
    "github.com/caddyserver/caddy/v2/caddyconfig/caddyfile"
    "github.com/caddyserver/caddy/v2/modules/caddyhttp"
)

func init() {
    caddy.RegisterModule(StatefulRouter{})
}

type StatefulRouter struct {
    Backends []string `json:"backends,omitempty"`
    sessions sync.Map
}

func (StatefulRouter) CaddyModule() caddy.ModuleInfo {
    return caddy.ModuleInfo{
        ID:  "http.handlers.stateful_routing",
        New: func() caddy.Module { return new(StatefulRouter) },
    }
}

func (s *StatefulRouter) Provision(ctx caddy.Context) error {
    // Any setup logic here
    return nil
}

func (s *StatefulRouter) Validate() error {
    // Validation logic here
    return nil
}

func (s *StatefulRouter) ServeHTTP(w http.ResponseWriter, r *http.Request, next caddyhttp.Handler) error {
    clientID := getClientID(r)
    backend, _ := s.sessions.LoadOrStore(clientID, s.selectBackend())
    backendStr := backend.(string)

    // Here you would implement the actual proxying logic
    // For demonstration, we'll just set a header
    w.Header().Set("X-Backend", backendStr)

    return next.ServeHTTP(w, r)
}

func (s *StatefulRouter) UnmarshalCaddyfile(d *caddyfile.Dispenser) error {
    for d.Next() {
        if !d.NextArg() {
            return d.ArgErr()
        }
        s.Backends = append(s.Backends, d.Val())
    }
    return nil
}

func (s *StatefulRouter) selectBackend() string {
    // Simple round-robin for demonstration
    return s.Backends[0] // In real implementation, you'd rotate through backends
}

func getClientID(r *http.Request) string {
    // For demonstration, we'll use the remote address
    return r.RemoteAddr
}
