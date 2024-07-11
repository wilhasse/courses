# Course

Link: https://www.boot.dev 

Docker Course   

## Ch1 Install

Debian  
https://docs.docker.com/desktop/install/debian/

## Ch2 Command Line

```bsah
# run docker: getting started
docker run -d -p 80:80 docker/getting-started:latest
# list running container
docker ps
# vist http://localhost

# stop container: get id from docker ps
docker stop CONTAINER_ID
# force stop container
docker kill CONTAINER_ID
# see all downloaded images
docker images
# executes command ls inside an container
docker exec CONTAINER_ID ls
# give an shell
docker exec -it CONTAINER_ID /bin/sh
# see stats of all container like top
docker stats
```
