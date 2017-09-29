# Parallel solver in Cuda


#The program ca be run in an Ubuntu terminal with the next command line: 

nvcc main.cpp hello_fn.cu -arch=sm_30 -lcusparse -lcublas -lcurand -o bicgra