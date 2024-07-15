# Course

Ardan Labs  
Jérôme Petazzoni  
https://2022-11-live.container.training/docker.yml.html

# Day 1

## Day 1 - Basic

```bash
# run detached
docker run -d jpetazzo/clock
# run read only
docker run -it --read-only jpetazzo/clock
# logs
docker logs CONTAINER_ID
# logs tail
docker logs CONTAINER_ID -f --tail 3
# stop container
docker stop CONTAINER_ID
# kill container
docker kill CONTAINER_ID
# start container
docker start CONTAINER_ID
# attach an shell
docker exec -it CONTAINER_ID /bin/bash
# kill all container
docker kill $(docker ps -q)
```

## Day 1 - Images

```bash
# list all images
docker images
# search remote registry
docker search ubuntu
# download image
docker pull ubuntu
# diff comparing to base image
docker diff CONTAINER_ID
# commit change made to image
docker commit CONTAINER_ID
# tagging images
docker tag NEW_IMAGE_ID figlet
# run using tag
docker run -it figlet
```

## Day 1 - Dockerfile

```bash
mkdir myimage
cd myimage
```

Plain Txt

```dockerFile
FROM ubuntu
RUN apt-get update
RUN apt-get install figlet
```

```bash
# build the image
docker build -t figlet .
# see history (layers)
docker history figlet
```

JSON

```dockerFile
RUN ["apt-get", "install", "figlet"]
```

Copy file

```dockerFile
FROM ubuntu
RUN apt-get update
RUN apt-get install -y build-essential
COPY hello.c /
RUN make hello
CMD /hello
```

## Day 1 - Go Exercise

Run CMD directly uses shell

```dockerFile
FROM golang
COPY . .
RUN go build dispatcher.go
CMD ./dispatcher
```

```bash
docker build . -t web
docker run web
docker exec 90831b3c48b0 ps aux
#USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
#root           1  0.0  0.0   2576   872 ?        Ss   13:29   0:00 /bin/sh -c ./dispatcher
#root           7  0.0  0.2 1600508 5764 ?        Sl   13:29   0:00 ./dispatcher
```

Correct way to invoke go binary

```dockerFile
FROM golang
COPY . .
RUN go build dispatcher.go
CMD ["./dispatcher"]
```

```bash
docker exec 2b4238a071b6 ps aux
#USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
#root           1  0.0  0.3 1600508 7884 ?        Ssl  13:30   0:00 ./dispatcher
```

# Day 2

## Day 2 - Network

```bash
# publish all ports
docker run -d -P nginx
# docker ps to find mapped port
docker ps
# or by container id
docker port CONTAINER_ID 80
# choose a specif port
docker run -d -p 8000:80 nginx
# find ip address
sudo docker inspect --format '{{ .NetworkSettings.IPAddress }}' CONTAINER_ID
```

## Day 2 - Development Workflow

Testing an example

```bash
# clone project
git clone https://github.com/jpetazzo/namer
# build an docker image
docker build -t namer .
# run exposing ports
docker run -dP namer
# check port
docker ps -l
```

Mapping source code to a local dir

```bash
# run cointaner with source code (src) mapped to local current dir
docker run --mount=type=bind,source=$(pwd),target=/src -dP namer
# edit company_name_generator.rb outside the cointaner
# change the color line 13 (color: royalblue;)
vim company_name_generator.rb
# find the port
docker ps
# see new color on the browser
http://localhost:32768/
```

## Day 2 - Network

```bash
# list networks
docker network ls
# create network
docker network create dev
docker network create prod
# create container with network
# each network trunk and prod is isolated
docker run --net trunk -ti alpine
docker run --net prod -ti alpine
# alias is a dns name
docker run --net prod --net-alias api -d nginx
ping api
```

## Day 2 - Compose

```bash
# install and run containers
docker-compose up
# start containers in the background
docker-compose up -d
# list all containers
docker-compose ps
# show logs
docker-compose logs
# stop all containers
docker-compose stop
# stop and remove
docker-compose down
```

# Day 3
