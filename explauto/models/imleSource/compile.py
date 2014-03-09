import subprocess
import argparse
import sys
import os

from string import Template

# template_files = ('main-python.cpp', 'myimle.h')
template_files = ('myimle.h', 'CMakeLists.txt', 'main-python.cpp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('in_ndims', type=int)
    parser.add_argument('out_ndims',type=int)

    args = parser.parse_args()

    d = args.in_ndims
    D = args.out_ndims
    fullname = '{}_{}'.format(d, D)

    build_path = os.path.join('build', fullname)
    src_path = os.path.join('python')

    if os.path.exists(build_path):
        print 'The dir {} already exists, clean it first.'.format(build_path)
        sys.exit(1)

    os.makedirs(build_path)

    for template in template_files:
	print template
        with open(os.path.join(src_path, template + '.tpl'), 'r') as f:
	    print 'f = ', f
            s = Template(f.read()).substitute(d=d, D=D, name=fullname)

        with open(os.path.join(src_path, template), 'w') as f:
            f.write(s)

    os.chdir(build_path)

    subprocess.call(["ccmake", "../.."])

    subprocess.call(["make"])

    print os.getcwd()
