import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))

from models.learner import *

for key, value in fmodels:
    desc = value.desc
    

fmodels = "Available forward models : \n{}\n".format('\n'.join("  {}{} : {}".format(key, (8-len(key))*' ', value.desc) for key, value in fwdclass.items()))
imodels = "Available inverse models : \n{}\n".format('\n'.join("  {}{} : {}".format(key, (8-len(key))*' ', value.desc) for key, value in invclass.items()))

print fmodels, imodels