from com.android.monkeyrunner import MonkeyRunner, MonkeyDevice
import sys
from time import sleep

# Note: this is a Monkeyrunner script which runs in Jython and so is therefore based on Python 2.7

device = MonkeyRunner.waitForConnection()

f = open(sys.argv[1])
try:
    for line in f:
        string, score, inputs = line.split('\t')
        print 'Inputting', string, 'for', score, 'points.'
        for x, y in eval(inputs):
            device.touch(x, y, 'DOWN_AND_UP')
            sleep(0.075)
finally:
    f.close()