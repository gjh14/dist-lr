sudo apt-get install cmake
sudo apt-get install protobuf-compiler
sudo apt-get install libprotobuf-dev
sudo apt-get install iftop

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
mkdir digit
wget -O ./digit/train.csv "https://storage.googleapis.com/kaggle-competitions-data/kaggle/3004/train.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1526388629&Signature=Vg3m7bEl2OS5%2Bxim%2FAPeJ%2FWfETPaNV%2Fn51Sc2bkaoKNFxb0d0P1XQ7GOXmplXgFYbgP5egw%2F%2FikMxzCOvuUyRDYDIVrViBgJ684QV%2FU6NzpfTMa6YlFrg1MInveTVGhLbRudmT9Y%2BJbFgx3LhARJPWV8MNLxzDCSnzJFUsKHWR64RGtNpPo6s3fBPp5J7lqSP9rELznnShrxycWmmPfEvpjMoxMhGmlygGVS962F4MBqXqzhu%2B6iJSlii0o%2FYOGpcs0mKc8LUsATJgfpqz2%2B81wNjBfQVz71FpyfJPljKx8BwBTE%2FDPkEgud6Y8Q5ctPg%2B6zvlwPEn3xhwfK3oattA%3D%3D"
python gen_data.py
python gen2.py

