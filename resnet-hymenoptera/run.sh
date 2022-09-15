echo -e "\033[36;1m ReBuild C++! \033[0m"
rm -rf build
mkdir build
cd build
cmake ..
make -j8
echo -e "\033[36;1m ReBuild Finished! \033[0m"

echo -e "\033[36;1m Runing C++ project! \033[0m"
starttime_cpp=`date +'%Y-%m-%d %H:%M:%S'`
./resnet
endtime_cpp=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds_cpp=$(date --date="$starttime_cpp" +%s); 
end_seconds_cpp=$(date --date="$endtime_cpp" +%s); 
echo -e "\033[36;1m C++ Time Used: $((end_seconds_cpp-start_seconds_cpp)) \033[0m"

echo -e "\033[36;1m Runing Python project! \033[0m"
cd ../python-src
starttime_python=`date +'%Y-%m-%d %H:%M:%S'`
python main.py
endtime_python=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds_python=$(date --date="$starttime_python" +%s); 
end_seconds_python=$(date --date="$endtime_python" +%s); 
echo -e "\033[36;1m Python Time Used: $((end_seconds_python-start_seconds_python)) \033[0m"
