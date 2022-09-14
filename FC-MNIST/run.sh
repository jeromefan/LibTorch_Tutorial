echo -e "\033[36;1m ReBuild C++! \033[0m"
rm -rf build
mkdir build
cd build
cmake ..
make -j8
echo -e "\033[36;1m ReBuild Finished! \033[0m"
echo -e "\033[36;1m Runing C++ project! \033[0m"
./FCnet
echo -e "\033[36;1m Runing Python project! \033[0m"
cd ../python-src
python main.py