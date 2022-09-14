rm -rf build
mkdir build
cd build
cmake ..
make -j8
echo -e "\033[36;1m ReBuild Finished! \033[0m"
./FCnet
echo -e "\033[36;1m Run Finished! \033[0m"