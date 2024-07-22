# Course

Ardan Labs  
Jérôme Petazzoni  
https://2022-11-live.container.training/kube.yml.html

# Day 1

## Day 1 - Kind (Kubernetes in Docker)

https://kind.sigs.k8s.io/

```bash
# kind
 go install sigs.k8s.io/kind@v0.23.0

# kubctl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"
echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# k9s
go install github.com/derailed/k9s@latest

# create cluster
kind create cluster

# create container
kubectl create deployment web --image=nginx

# expose port
kubectl expose deploy web --port=80

# see the services
kubectl get svc

# expose outside the container (not network)
kubectl port-forward service/web 8080:80
telnet localhost 8080
```

Custom port cluster

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 80
        hostPort: 80
```

```bash
kind delete cluster
kind create cluster --config kind-80.yaml
kubectl get nodes
docker ps
```

Helm (Kubernetes Package Manager)

```bash
# install
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
# add repo
helm repo add traefik https://helm.traefik.io/traefik
helm repo update
```

Traefik (proxy load balancer)
Expose port 80

```bash
# no port (Connection reset by peer)
curl localhost
helm upgrade  --install traefik --namespace traefik --create-namespace traefik/traefik
kubectl get nodes -o wide
# take not of INTERNAL-IP in this case 172.18.0.2
kubectl edit svc -n traekit traekit
# add in spec after clusterIPs
# externalIPs
# - 172.18.1.2
# save and exit
# curl now returns 404 page not found
curl localhost
```

Deploy nginx

```bash
kubectl create deploy web --image=nginx
kubectl expose deploy web --port=80
kubectl create ingress web --rule=nginx.localtest.me/*=web:80
curl nginx.localtest.me
```

## Day 1 - kubectl Basics

```bash
# check nodes (normal and more verbose)
kubectl get nodes
kubectl get nodes -v6
kubectl get nodes -v9
# verbose json / yaml
kubectl get nodes -o json
kubectl get nodes -o yaml
# extract specific info with jq
kubectl get nodes -o json |
        jq ".items[] | {name:.metadata.name} + .status.capacity"

# describe node
kubectl describe node kind-worker

# list namespaces services
kubectl get namespaces
kubectl get services

# list pods specific namespace
kubectl get pods --namespace kube-system -o wide
```

## Day 1 - Run containers

```bash
# run ping
kubectl run pingpong --image alpine ping 127.0.0.1
# see logs
kubectl logs pingpong
# last log and continue
kubectl logs pingpong --tail 1 --follow
# scale (error could not find the resources)
kubectl scale pod pingpong --replicas=3 -v6
```

## Day 1 - Scale containers

```bash
# see all pods
watch kubectl get pods
# create a new pod
kubectl create deployment pingpong --image alpine -- ping localhost
# more information
kubectl get all
# scale replicasets
kubectl scale deployment pingpong --replicas 3
# remove original pod
kubectl delete pod pingpong
# try to remove an replicaset (it will create another to keep 3 instances)
kubectl delete pod pingpong-c855654bb-lztnq
```
