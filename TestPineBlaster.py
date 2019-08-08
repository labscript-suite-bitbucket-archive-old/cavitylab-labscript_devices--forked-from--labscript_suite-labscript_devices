import sys
sys.path.append("C:\\labscript\\user_scripts\\labscriptlib\\RbCavity")
from connectiontable import do_connectiontable
from openclosebeam import *
from SetUpLab import *
#from LoadMOT import *
import ntpath

do_connectiontable()
start()
t = 0
t = SetUpLab(t)

ScopeTrigger.go_high(t)

t+=1e-3

wait('my_first_wait', t=t, timeout=5)

t+=4e-6

ScopeTrigger.go_low(t)

t+=800e-3
#
ScopeTrigger.go_high(t)
#
#t+=1e-3
#
#wait('my_second_wait', t=t, timeout=5)
#
#t+=4e-6
#
#ScopeTrigger.go_low(t)

t+=1e-3

t = SetUpLab(t)

t+=1e-3

stop(t+4e-3)
