# Course

Link: https://www.boot.dev

Docker Course

## Ch1 Install

Debian  
https://docs.docker.com/desktop/install/debian/

Enable user to run docker

```bash
# add user to docker group
sudo usermod -aG docker cslog
# logout login
exit
# check if docker is in the group
groups cslog
# test
docker run hello-world
```

## Ch2 Command Line

```bash
# run hello world
docker run busybox echo hello world
# run docker: getting started
docker run -d -p 80:80 docker/getting-started:latest
# list running container
docker ps -a
# vist http://localhost

# stop container: get id from docker ps
docker stop CONTAINER_ID
# force stop container
docker kill CONTAINER_ID
# restart container
docker restart CONTAINER_ID
# see all downloaded images
docker images
# executes command ls inside an container
docker exec CONTAINER_ID ls
# give an shell
docker exec -it CONTAINER_ID /bin/sh
# see stats of all container like top
docker stats
```

## Ch3 Storage

```bash
# create a new volume
volume create ghost-vol
# list volumes
docker volume ls
# inspect one volume
docker inspect ghost-vol
# remove volume
docker rm ghost-vol
```

## Ch4 Network

```bash
# no network to an container
docker run -d --network none docker/getting-started
# pull caddy
docker pull caddy
# create network
docker network create caddytest
# containers sharing the same network
# html1
#<html><body>
#    <h1>Hello from server 1</h1>
#</body></html>
docker run --network caddytest --name caddy1 -p 8001:80 -v $PWD/index1.html:/usr/share/caddy/index.html caddy
# html2
#<html><body>
#    <h1>Hello from server 2</h1>
#</body></html>
docker run --network caddytest --name caddy2 -p 8002:80 -v $PWD/index2.html:/usr/share/caddy/index.html caddy
# load balancer
# configuring: create an file Caddyfile
#localhost:80
#
#reverse_proxy caddy1:80 caddy2:80 {
#        lb_policy round_robin
#}
docker run --network caddytest -p 8080:80 -v $PWD/Caddyfile:/etc/caddy/Caddyfile caddy
```

## Ch5 Dockerfiles

```dockerfile
# This is a comment

# Use a lightweight debian os
# as the base image
FROM debian:stable-slim

# execute the 'echo "hello world"'
# command when the container runs
CMD ["echo", "hello world"]
```

```bash
docker build . -t helloworld:latest
docker run helloworld
```

Run httpgo (own binary) inside linux docker

```go
package main

import (
	"fmt"
	"log"
	"net/http"
	"time"
)

func main() {
	m := http.NewServeMux()

	m.HandleFunc("/", handlePage)

	const addr = ":8080"
	srv := http.Server{
		Handler:      m,
		Addr:         addr,
		WriteTimeout: 30 * time.Second,
		ReadTimeout:  30 * time.Second,
	}

	// this blocks forever, until the server
	// has an unrecoverable error
	fmt.Println("server started on ", addr)
	err := srv.ListenAndServe()
	log.Fatal(err)
}

func handlePage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	w.WriteHeader(200)
	const page = `<html>
<head></head>
<body>
	<p> Hello from Docker! I'm a Go server. </p>
</body>
</html>
`
	w.Write([]byte(page))
}
```

```dockerfile
FROM debian:stable-slim

# COPY source destination
COPY httpgo /bin/httpgo

CMD ["/bin/httpgo"]
```

```bash
docker build . -t httpgo:latest
docker run -p 8080:8080 httpgo
```

## Ch6 Publish

```bash
# push
docker build . -t USERNAME/httpgo
docker run -p 8080:8080 USERNAME/httpgo
docker push USERNAME/httpgo

# remove local container and pull from docker
docker image rm USERNAME/httpgo
docker run -p 8080:8080 USERNAME/httpgo

# new version (tag)
docker build . -t USERNAME/httpgo:0.2.0
docker run -p 8080:8080 USERNAME/httpgo:0.2.0
docker push USERNAME/httpgo:0.2.0
```

## dhcpd example

```bash
# only run
docker run -it --rm --init --net host -v "/docker/dhcpd/data/data":/data networkboot/dhcpd eth0

# restart always
docker run --restart always --net host -d -v "/docker/dhcpd/data":/data networkboot/dhcpd vmbr0

# check policy 
docker ps
docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' 15bfb3a1f62d

# check docker service
systemctl status docker
```
