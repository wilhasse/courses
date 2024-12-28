# Introduction

Facebook Cachelib   
https://cachelib.org

# Build

Ennvironment: Rocky Linux 9.4  
https://rockylinux.org/pt-BR/download  

Install Fast Float

```bash
git clone https://github.com/fastfloat/fast_float.git
cd fast_float
mkdir build
cd build
cmake ..
sudo make install
```

Install XXhash Devel

```bash
sudo dnf install xxhash-devel
```

Issue with 

Edit contrib/build-package.sh
add external_git_tag the last version that compiled in Rocky Linux

```bash
  fbthrift)
    NAME=fbthrift
    SRCDIR=cachelib/external/$NAME
    update_submodules=yes
    external_git_tag="v2023.01.09.00"
    cmake_custom_params="-DBUILD_SHARED_LIBS=ON"
    ;;
```

Compile

Options:  

```plain
  -d    build with DEBUG configuration  
  -j    build using all available CPUs ('make -j')  
  -v    verbose build  
```

```bash
./contrib/build.sh -d -j -v
```

After first time add option to skip package instalation
-O    skip OS package installation (apt/yum/dnf)
