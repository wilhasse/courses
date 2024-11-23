# Guide

https://doc.polardbx.com/en/quickstart/topics/quickstart-pxd-cluster.html

# Install

Pxd: Follow
https://doc.polardbx.com/en/quickstart/topics/quickstart.html  

Prepare 3 machines (VM)
Debian 12

Install Docker (version >= 18.09)

```bash
# Update package index and install prerequisites
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Docker version 27.3.1, build ce12230
docker --version
```

Add permission to user

```bash
sudo usermod -aG docker $USER
newgrp docker

# Test
docker ps
```

SSH without password from pxd machine

```bash
ssh-keygen -t rsa
ssh-copy-id 10.1.1.132
ssh-copy-id 10.1.1.121
ssh-copy-id 10.1.1.129
```

# Deploy

Topoloy Yaml file example

```yaml
version: v1
type: polardbx
cluster:
  name: pxc_test
  gms:
    image: polardbx/polardbx-engine:v2.4.0_8.4.19
    host_group: [10.1.1.132]
  cn:
    image: polardbx/polardbx-sql:v2.4.0_5.4.19
    replica: 2
    nodes:
      - host: 10.1.1.121
      - host: 10.1.1.129
    resources:
      mem_limit: 4G
  dn:
    image: polardbx/polardbx-engine:v2.4.0_8.4.19
    replica: 2
    nodes:
      - host_group: [10.1.1.121]
      - host_group: [10.1.1.129]
    resources:
      mem_limit: 4G
  cdc:
    image: polardbx/polardbx-cdc:v2.4.0_5.4.19
    replica: 1
    nodes:
      - host: 10.1.1.132
    resources:
      mem_limit: 4G
```

# Test Docker API

In all nodes check Docker API response  
Example: curl http://10.1.1.121:2375/version  

If returns:  
curl: (7) Failed to connect to 10.1.1.132 port 2375 after 0 ms: Couldn't connect to server

Enable API:  
Create the systemd override directory and file:

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d/
sudo nano /etc/systemd/system/docker.service.d/override.conf
```

Add this content to the override.conf file:

```bash
iniCopy[Service]
ExecStart=
ExecStart=/usr/bin/dockerd -H fd:// -H tcp://0.0.0.0:2375 --containerd=/run/containerd/containerd.sock
```

Set proper permissions:

```bash
sudo chmod 644 /etc/systemd/system/docker.service.d/override.conf
```

Reload systemd and restart Docker:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

Verify it's working:

```bash
# Test local connection
curl http://localhost:2375/version

# Test remote connection (from another machine)
curl http://REMOTE_IP:2375/version
```

Change python how authenticate to API Docker
sudo pico /home/cslog/venv/lib/python3.8/site-packages/deployer/core/docker_manager.py

From:
```bash
def get_client(host):
    docker_url = "ssh://%s:%d" % (host, Config.ssh_port())        
    client = docker.DockerClient(base_url=docker_url, timeout=60, max_pool_size=100)
    return client
```
To:
```bash
def get_client(host):
    # Change from SSH to TCP
    docker_url = f"tcp://{host}:2375"
    client = docker.DockerClient(base_url=docker_url, timeout=60, max_pool_size=100)
    return client
```

Finally create the cluster:

```bash
pxd create -file polardbx.yaml
yaml file: polardbx.yaml
Processing  [------------------------------------]    0%    pre check
Processing  [##----------------------------------]    7%    generate topology
Processing  [#####-------------------------------]   14%    check docker engine version
Processing  [#######-----------------------------]   21%    pull images
Processing  [##########--------------------------]   28%    create gms node
Processing  [############------------------------]   35%    create gms db and tables
Processing  [###############---------------------]   42%    create PolarDB-X root account
Processing  [##################------------------]   50%    create dn
Processing  [####################----------------]   57%    register dn to gms
Processing  [#######################-------------]   64%    create cn
Processing  [#########################-----------]   71%    wait cn ready
Processing  [############################--------]   78%    create cdc containers
Processing  [##############################------]   85%    create columnar containers
Processing  [#################################---]   92%    wait PolarDB-X ready
Processing  [####################################]  100%
PolarDB-X cluster create successfully, you can try it out now.
Connect PolarDB-X using the following command:

    mysql -h10.1.1.121 -P57780 -upolardbx_root -pVVbaOLta
    mysql -h10.1.1.129 -P61620 -upolardbx_root -pVVbaOLta
```
