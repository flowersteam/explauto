import sys

# turning printing on and off  
class NullDevice():
    def write(self, s):
        pass

stdout_bak = sys.stdout
stdout_null = NullDevice()
        
def turn_off_print():
    sys.stdout = stdout_null

def turn_on_print():
    sys.stdout = stdout_bak
