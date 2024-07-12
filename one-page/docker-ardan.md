# Course

Ardan Labs  
Jérôme Petazzoni  
https://2022-11-live.container.training/docker.yml.html

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
