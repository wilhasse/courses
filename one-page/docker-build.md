# Docker

## Dockerfile

Example

```dockerfile
FROM debian:bookworm-20240926-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Sao_Paulo
RUN apt-get update && apt-get install -y \
    openssh-server \
    subversion \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 22
ADD entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]
```

Entrypoint

```dockerfile
#!/bin/bash
set -e

# Create required directories
mkdir -p /var/run/sshd

# Set root password if provided via environment variable
if [ ! -z "$ROOT_PASSWORD" ]; then
    echo "root:${ROOT_PASSWORD}" | chpasswd
fi

# Start SSH service
echo "Starting SSH server..."
exec /usr/sbin/sshd -D
```

## Build

Create Dockerfile
Build image on Windows
Install and run Docker desktop

Generate image

```bash
# build
docker build --no-cache -t tisvn:0.2.0 .
```

Save image

```bash
# windows
# avoid redirecting to file 
# docker save image:0.2.0 > image.tar
# linux error importing: archive/tar: invalid tar header
# use option -
docker save image:0.2.0 -o image.tar
```

Export

```bash
docker create --name temp_douradina douradina:0.2.0
docker export temp_douradina | gzip > douradina.tgz
docker rm temp_douradina
```

## Import

From save

```bash
# copy image
scp ...
# load image
docker load -i douradina.tar
```

From export

```bash
zcat douradina.tgz | docker import - douradina:0.2.0
```

## Run

```bash
docker run -d --network=cslog -p 22:2224 -e ROOT_PASSWORD="teste" --restart=unless-stopped --name douradina douradina:0.2.0 /entrypoint.sh
```

## Other commands

```bash
# stop container
docker stop container_id

# remove a container
docker rm container_id

# list all running container 
docker ps

# list all
docker ps -a

# starts a container listed in ps as exited
docker start container_id

# see what happened to the container
docker logs container_id

# execute a command
docker exec -it container_id /bin/bash

# list images
docker images

# remove an image
# docker rmi percona/percona-server:8.4.2-2.1
docker rmi image:version
```