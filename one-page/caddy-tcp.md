# Caddy

Caddy doesn't work proxy TCP only HTTP

For TCP proxy/load balancer, use this plugin Layer 4:  
https://github.com/mholt/caddy-l4

# Build

Use xcaddy to compile this plugin into Caddy

```bash
# First install Go if you haven't already
# Then install xcaddy
go install github.com/caddyserver/xcaddy/cmd/xcaddy@latest

# Build Caddy with the layer4 plugin
xcaddy build --with github.com/mholt/caddy-l4
```

## Verify plugin

```bash
caddy list-modules | findstr layer4

Standard modules: 121
caddy.listeners.layer4
layer4
layer4.handlers.echo
layer4.handlers.proxy
layer4.handlers.proxy_protocol
layer4.handlers.socks5
layer4.handlers.subroute
layer4.handlers.tee
layer4.handlers.throttle
layer4.handlers.tls
layer4.matchers.clock
layer4.matchers.dns
layer4.matchers.http
layer4.matchers.local_ip
layer4.matchers.not
layer4.matchers.openvpn
layer4.matchers.postgres
layer4.matchers.proxy_protocol
layer4.matchers.quic
layer4.matchers.rdp
layer4.matchers.regexp
layer4.matchers.remote_ip
layer4.matchers.socks4
layer4.matchers.socks5
layer4.matchers.ssh
layer4.matchers.tls
layer4.matchers.winbox
layer4.matchers.wireguard
layer4.matchers.xmpp
layer4.proxy.selection_policies.first
layer4.proxy.selection_policies.ip_hash
layer4.proxy.selection_policies.least_conn
layer4.proxy.selection_policies.random
layer4.proxy.selection_policies.random_choose
layer4.proxy.selection_policies.round_robin
```

## JSON config

```json
{
  "logging": {
    "logs": {
      "default": {
        "writer": {
          "output": "file",
          "filename": "C:/Caddy/caddy.log"
        },
        "level": "INFO",
        "encoder": {
          "format": "console"
        }
      }
    }
  },
  "apps": {
    "layer4": {
      "servers": {
        "example": {
          "listen": [":8209"],
          "routes": [
            {
              "handle": [
                {
                  "handler": "proxy",
                  "upstreams": [
                    {"dial": ["localhost:8230"]},
                    {"dial": ["localhost:8231"]},
                    {"dial": ["localhost:8232"]},
                    {"dial": ["localhost:8233"]},
                    {"dial": ["localhost:8234"]},
                    {"dial": ["localhost:8235"]},
                    {"dial": ["localhost:8236"]},
                    {"dial": ["localhost:8237"]},
                    {"dial": ["localhost:8238"]},
                    {"dial": ["localhost:8239"]}
                  ]
                }
              ]
            }
          ]
        }
      }
    }
  }
}
```

## Install Windows

```bash
sc.exe create Caddy binPath= "\"C:\Caddy\caddy.exe\" run --config \"C:\Caddy\config.json\""
```