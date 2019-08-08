#####################################################################
#                                                                   #
# /SpectrumM4X6620.py                                               #
#                                                                   #
#                                                                   #
#####################################################################

#############
#
#############

from labscript_devices import runviewer_parser, labscript_device, BLACS_tab, BLACS_worker
from labscript import IntermediateDevice, Device, config, LabscriptError, StaticAnalogQuantity, AnalogOut, DigitalOut
#from labscript_utils.unitconversions import NovaTechDDS9mFreqConversion, NovaTechDDS9mAmpConversion
from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED
from blacs.device_base_class import DeviceTab
from pyspcm import *
import numpy as np
import time
import math
from MCLController import MCLController
from scipy.fftpack import fft, rfft, irfft, ifft
from scipy.signal import chirp
import labscript_utils.h5_lock, h5py
from ctypes import *
import struct

class MirrorParams():
    def __init__(self,InAngle1,InAngle2,OutAngle1,OutAngle2,Length):
        self.InAngle1 = InAngle1
        self.InAngle2 = InAngle2
        self.OutAngle1 = OutAngle1
        self.OutAngle2 = OutAngle2
        self.Length = Length

@labscript_device
class CavityMirrors(Device):

    def __init__(self,name,parent_device):
        self.BLACS_connection = 5
        Device.__init__(self,name,parent_device,connection=self.BLACS_connection)

    def reset(self, t): # This doesn't do anything but must be here.
        return t

    # Load profile table containing data into h5 file
    def generate_code(self, hdf5_file):
        device = hdf5_file.create_group('/devices/' + self.name)

    def stop(self):
        return


@BLACS_tab
class CavityMirrorsTab(DeviceTab):
    def initialise_GUI(self):
        # Capabilities
        self.num_axes = 5
        self.base_units = '?'
        self.base_min = 0.0
        self.base_max = 3.0
        self.base_step = 0.001
        self.base_decimals = 3

        # Create the axes output objects
        axes_prop = {}
        axis_names = ['In Horiz', 'In Vert', 'Out Horiz', 'Out Vert', 'Length']
        for i in range(self.num_axes):
            axes_prop[axis_names[i]] = {'base_unit':self.base_units,
                                   'min':self.base_min,
                                   'max':self.base_max,
                                   'step':self.base_step,
                                   'decimals':self.base_decimals
                                  }

        # Create the output objects
        self.create_analog_outputs(axes_prop)
        # Create widgets for analog outputs only
        dds_widgets,axes_widgets,do_widgets = self.auto_create_widgets()

        # and auto place the widgets in the UI
        self.auto_place_widgets(("Mirror Axes",axes_widgets))

        # Create and set the primary worker
        self.create_worker("main_worker",CavityMirrorsWorker,{})
        self.primary_worker = "main_worker"

        # Set the capabilities of this device
        self.supports_remote_value_check(False)
        self.supports_smart_programming(False)


@BLACS_worker
class CavityMirrorsWorker(Worker):
    def init(self):
        self.inputStage = MCLController(3108)
        self.outputStage = MCLController(3109)
        return

### These functions don't currently do anything useful, but they could be used to disply useful info on GUI ###
    def check_remote_values(self):
        return

    def setCavityParams(self,MirrorParamObj):
        self.outputStage.writeZPosition(MirrorParamObj.Length)
        self.outputStage.writeAnglePosition(1, MirrorParamObj.OutAngle1)
        self.outputStage.writeAnglePosition(2, MirrorParamObj.OutAngle2)
        self.inputStage.writeAnglePosition(1, MirrorParamObj.InAngle1)
        self.inputStage.writeAnglePosition(2, MirrorParamObj.InAngle2)
        print "Z Pos:\t", MirrorParamObj.Length, "\t\tActual Z Pos:\t", self.outputStage.readZPosition()
        print "Out V:\t", MirrorParamObj.OutAngle1, "\t\tActual Out V:\t", self.outputStage.readAnglePosition(1)
        print "Out H:\t", MirrorParamObj.OutAngle2, "\t\tActual Out H:\t", self.outputStage.readAnglePosition(2)
        print "In V:\t", MirrorParamObj.InAngle1, "\t\tActual In V:\t", self.inputStage.readAnglePosition(1)
        print "In H:\t", MirrorParamObj.InAngle2, "\t\tActual In H:\t", self.inputStage.readAnglePosition(2)

    def program_manual(self,front_panel_values):
        MirrorParamObj = MirrorParams(front_panel_values['In Vert'],front_panel_values['In Horiz'],front_panel_values['Out Vert'],front_panel_values['Out Horiz'],front_panel_values['Length'])
        self.setCavityParams(MirrorParamObj)
        print('hello')
        print(front_panel_values)
        return self.check_remote_values()

    def transition_to_buffered(self,device_name,h5file,initial_values,fresh):
        return

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def abort_buffered(self):
        return self.transition_to_manual(True)

    def transition_to_manual(self,abort = False):
        if abort:
            self.shutdown()
            self.init()
        return True

    def shutdown(self):
        self.inputStage.releaseHandle()
        self.outputStage.releaseHandle()
        return
