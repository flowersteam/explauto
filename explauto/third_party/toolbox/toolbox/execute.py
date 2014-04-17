from __future__ import print_function
import subprocess

c_end     = '\033[0m'
c_cyan    = '\033[0;36m'

def execute(cmd, print_cmd=True, print_output=False):
    if print_cmd:
        print(cmd)
    try:
        proc = subprocess.Popen(cmd,  stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                      shell=True)

        stdout, stderr = proc.communicate()
        if print_output:
            print(c_cyan+stdout+c_end)
        return stdout
    except OSError as e:
        print(cmd)
        raise e
