g++ -o featextract.so -fPIC -shared featextract.cpp \
	-I/usr/include/python3.5m \
	-L/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
	-lpython3.5  \
	-I/usr/local/boost_1_72_0_py3/include \
	-L/usr/local/boost_1_72_0_py3/lib \
	-lboost_python35 -lboost_numpy35
