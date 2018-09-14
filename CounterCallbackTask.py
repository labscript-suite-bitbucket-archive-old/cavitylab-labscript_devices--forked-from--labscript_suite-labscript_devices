# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 18:46:28 2016

@author: Stanford University
"""
from PyDAQmx import *
import numpy as np

# Use this class if you want the counter to auto-register every N samples
class CounterCallbackTask(Task):
    def __init__(self, counter_chnl, clk_chnl, sample_rate, samps_per_acq):
        ## TODO: If we have different samps_per_acq in one experiment, we will have to take the smallest and format the allData property accordingly.
        Task.__init__(self)
        self.counter_chnl = counter_chnl
        self.clk_chnl = clk_chnl
        self.sample_rate = sample_rate
        self.samps_per_acq = samps_per_acq
        
        self.data = np.zeros(self.samps_per_acq)
        self.allData = []
        self.CreateCICountEdgesChan(self.counter_chnl, '', DAQmx_Val_Rising, 0, DAQmx_Val_CountUp)
        self.CfgSampClkTiming(self.clk_chnl, self.sample_rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, self.samps_per_acq)
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.samps_per_acq, 0)
        self.AutoRegisterDoneEvent(0)
    def EveryNCallback(self):
        read = int32()
        self.ReadCounterF64(DAQmx_Val_Auto, 10, self.data, len(self.data), byref(read), None)
        self.allData.extend(self.data.tolist())
        return 0 # The function should return an integer
    def DoneCallback(self, status):
        return 0 # The function should return an integer

# Want to register when CPT finishes
class CPTTask(Task):
    def __init__(self, CPT_chnl, trig_chnl, sample_freq, samps_per_acq, count):
        Task.__init__(self)
        self.count = count
        self.status = "CPT task"
        self.CPT_chnl = CPT_chnl
        self.trig_chnl = trig_chnl 
        self.sample_freq = sample_freq
        self.samps_per_acq = samps_per_acq
        
        self.CreateCOPulseChanFreq(self.CPT_chnl, '', DAQmx_Val_Hz, DAQmx_Val_Low, 0, self.sample_freq, 0.5)  
        self.CfgImplicitTiming(DAQmx_Val_FiniteSamps, self.samps_per_acq) 
        self.CfgDigEdgeStartTrig(self.trig_chnl, DAQmx_Val_Rising)
        self.AutoRegisterDoneEvent(0)
    def DoneCallback(self, status):
        self.status = "CPT task done"
        self.count += 1
        return 0 # The function should return an integer


#class TriggerCPT(Task):
#    def __init__(self):
#        Task.__init__(self)
#        self.counter_chnl = counter_chnl
#        self.clk_chnl = clk_chnl
#        self.sample_rate = sample_rate
#        self.samps_per_acq = samps_per_acq
#        
#        self.data = np.zeros(self.samps_per_acq)
#        self.allData = []
#        self.CreateCICountEdgesChan(self.counter_chnl, '', DAQmx_Val_Rising, 0, DAQmx_Val_CountUp)
#        self.CfgSampClkTiming(self.clk_chnl, self.sample_rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, self.samps_per_acq)
#        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.samps_per_acq, 0)
#        self.AutoRegisterDoneEvent(0)
#    def EveryNCallback(self):
#        read = int32()
#        self.ReadCounterF64(DAQmx_Val_Auto, 10, self.data, len(self.data), byref(read), None)
#        self.allData.extend(self.data.tolist())
#        return 0 # The function should return an integer
#    def DoneCallback(self, status):
#        return 0 # The function should return an integer

#class CallbackTask(Task):
#    def __init__(self):
#        Task.__init__(self)
#        self.data = np.zeros(100)
#        self.a = []
#        self.CreateCICountEdgesChan("/NIMultiDAQCard0/ctr0", '', DAQmx_Val_Rising, 0, DAQmx_Val_CountUp)
#        self.CfgSampClkTiming("/NIMultiDAQCard0/PFI13", 100, DAQmx_Val_Rising, DAQmx_Val_ContSamps, 100)
#        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer,100,0)
#        self.AutoRegisterDoneEvent(0)
#    def EveryNCallback(self):
#        read = int32()
#        self.ReadCounterF64(DAQmx_Val_Auto, 10, self.data, len(self.data), byref(read), None)
#        self.a.extend(self.data.tolist())
##        print self.data[0]
#        return 0 # The function should return an integer
#    def DoneCallback(self, status):
##        print "done" #print "Status",status.value
#        return 0 # The function should return an integer
    