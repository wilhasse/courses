# Repo

Link: https://github.com/wilhasse/zig-grpc-example  

Compiling: Zig 0.14.0-dev.1057

# Install

gRPC

```bash
sudo apt-get install libgrpc-dev
```

Bazel 

```bash
sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update
sudo apt install bazel -y
bazel
sudo apt update && sudo apt install bazel-5.1.0
```

Libatomic

```bash
ldconfig -p | grep libatomic
ls -l /usr/lib/x86_64-linux-gnu/libatomic*
/usr/sbin/ldconfig -p | grep libatomic
ls -l /usr/lib/x86_64-linux-gnu/libatomic*
sudo ln -s libatomic.so.1.2.0 libatomic.so
```

Clap

```bash
zig fetch --save git+https://github.com/Hejsil/zig-clap
```

Make

```bash
cslog@godev:~/zig-grpc-example$ make
bazel build //...
Starting local Bazel server and connecting to it...
INFO: Invocation ID: 08956226-0b52-44c9-b1d1-eb61a6e380fe
INFO: Analyzed 23 targets (126 packages loaded, 4236 targets configured).
INFO: Found 23 targets...
```

