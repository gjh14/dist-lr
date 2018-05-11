sudo apt-get install cmake
sudo apt-get install protobuf-compiler
sudo apt-get install libprotobuf-dev

wget https://github.com/zeromq/libzmq/releases/download/v4.2.3/zeromq-4.2.3.tar.gz
tar -xzvf zeromq-4.2.3.tar.gz
rm zeromq-4.2.3.tar.gz
cd zeromq-4.2.3
./configure
make -j4 
sudo make install
cd ..   

git submodule init
git clone https://github.com/ctliu3/ps-lite.git
mkdir build
cd build
cmake ..
make -j4 
cd ..

cd examples
python gen_data.py

