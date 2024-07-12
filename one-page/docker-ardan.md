# Course

Ardan Labs  
Jérôme Petazzoni  
https://2022-11-live.container.training/docker.yml.html

## Day 1

```bash
# run detached
docker run -d jpetazzo/clock
# logs
docker logs CONTAINER_ID
# logs tail
 docker logs CONTAINER_ID -f --tail 3
 # kill all container
 docker kill $(docker ps -q)
```

```bash
# list all images
docker images
```
