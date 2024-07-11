# Course

Link: https://www.boot.dev

Docker Course

## Ch1 Install

Debian  
https://docs.docker.com/desktop/install/debian/

## Ch2 Command Line

```bash
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
