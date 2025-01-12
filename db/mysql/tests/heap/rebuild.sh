cd ~/heap
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
make
