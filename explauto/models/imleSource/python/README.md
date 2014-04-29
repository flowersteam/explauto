# Python bindings for IMLE

* Dependency : boost.python, numpy

0. Copy the python directory, the compile.py and the CMakeLists.txt files into the imleSource directory. This will overwrite the existing CMakeLists.txt: another option is simply to add 'add_subdirectory( "python" )' at the end of it.
1. cd into the imleSource directory.
2. run 'python compile.py d D' where d and D are the input and output dimensions, respectively.
3. run 'python test.py' from imleSource/python (and look at the content of test.py). The test uses d=3 and D=2 (see point 2. above and compile accordingly)


