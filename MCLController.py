# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 12:48:54 2016

@author: Emily
"""
from ctypes import *
import time
import numpy as np

class MCLController:
    def __init__(self, serialNumber):
        self.mdll = cdll.Madlib
        self.mdll.MCL_GrabAllHandles()
        self.handle = self.mdll.MCL_GetHandleBySerial(c_short(serialNumber))
        self.visited = []
        self.x = self.readAnglePosition(1)
        self.y = self.readAnglePosition(2)
        if serialNumber == 3109:
            self.z = self.readZPosition()

    def readZPosition(self):
        zAxisRead = self.mdll.MCL_SingleReadZ
        zAxisRead.restype = c_double
        self.z = zAxisRead(self.handle)
        #print "Current z axis position: ", zAxisRead(self.handle)
        return self.z

    def readAnglePosition(self, axis):
        axisRead = self.mdll.MCL_SingleReadN
        axisRead.restype = c_double
        #print "Current axis {} position: ".format(axis), axisRead(c_uint(axis),self.handle)
        if axis == 1:
            self.x = axisRead(c_uint(axis),self.handle)
            return self.x
        elif axis == 2:
            self.y = axisRead(c_uint(axis),self.handle)
            return self.y

    def writeZPosition(self, position):
        if position > 250:
            print "You cannot set the z value greater than 250 microns."
            return
        #position = input("Where would you like to position the z axis (microns)?\n")
        zAxisWrite = self.mdll.MCL_SingleWriteZ(c_double(position), self.handle)
        #print "Z axis success? {}".format("Yes" if zAxisWrite == 0 else "No, error code: {} ".format(zAxisWrite))
        return

    def writeAnglePosition(self, axis, position):
        if position > 5:
            print "You cannot set the angle greater than 5 mradians."
            return
        angleAxisWrite = self.mdll.MCL_SingleWriteN(c_double(position),c_uint(axis), self.handle)
        #print "Axis {} write success? {}".format(axis, "Yes" if angleAxisWrite == 0 else "No, error code: {} ".format(angleAxisWrite))
        return

    def recordPosition(self):
        self.visited.append((self.x,self.y))
        return  (self.x,self.y)

    def wait(self, milliseconds):
        attached = self.mdll.MCL_DeviceAttached
        attached.restype = c_bool
        attached(c_uint(milliseconds), self.handle)
        #print "Waiting ", milliseconds, "ms"
        return

    def readZAndWait(self, position):
        test = 1
        while test:
            curPos = self.readZPosition()
            if curPos < position + .1 and curPos > position - .1:
                test = 0
            else:
                #print "waiting for stage to move..."
                continue
        return

    def readAngleAndWait(self, axis, position):
        test = 1
        while test:
            curPos = self.readAnglePosition(axis)
            if curPos < position + .001 and curPos > position - .001:
                test = 0
            else:
                #print "waiting for stage to move..."
                continue
        return

    def releaseHandle(self):
        self.mdll.MCL_ReleaseHandle(self.handle)
        return
