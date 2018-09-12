#####################################################################
#                                                                   #
# /AD9914.py                                                        #
#                                                                   #
#                                                                   #
#####################################################################
from labscript_devices import runviewer_parser, labscript_device, BLACS_tab, BLACS_worker

from labscript import IntermediateDevice, AD_DDS, Device, config, LabscriptError
#from labscript_utils.unitconversions import NovaTechDDS9mFreqConversion, NovaTechDDS9mAmpConversion

import numpy as np
import labscript_utils.h5_lock, h5py
from ctypes import *
import struct
import binascii
import datetime
import time


@labscript_device
class AD9914(IntermediateDevice):
    description = 'AD9914'
    allowed_children = [AD_DDS]
    clock_limit = 3.5e9

    def __init__(self, name, parent_device, com_port):
        IntermediateDevice.__init__(self, name, parent_device)
        self.BLACS_connection = com_port

    # def add_device(self, device): # !!!
    #     Device.add_device(self, device)
    #     # The Novatech doesn't support 0Hz output; set the default frequency of the DDS to 0.1 Hz:
    #     device.frequency.default_value = 0.1

    def generate_code(self, hdf5_file):

        IntermediateDevice.generate_code(self, hdf5_file)

        DDSProfs = {}

        if len(self.child_devices) > 1:
            raise LabscriptError("Too many child_devices. This device expects exactly 1 AD_DDS child output.")

        output = self.child_devices[0]

        if isinstance(output, AD_DDS):
            prefix = output.connection[0]
            channel = int(output.connection[1])
        else:
            raise Exception('Got unexpected device.')

        numDDSProfs = len(output.profiles)

        grp = hdf5_file.create_group('/devices/'+self.name)


        if numDDSProfs:
            profile_dtypes = [('freq%d'%i,np.float) for i in range(numDDSProfs)] + \
                            [('phase%d'%i,np.float) for i in range(numDDSProfs)] + \
                            [('amp%d'%i,np.float) for i in range(numDDSProfs)]


            profile_table = np.zeros(1, dtype=profile_dtypes)

            for profile in output.profiles:
                profile_table['freq'+profile[-1:]] = output.profiles[profile]['freq']
                profile_table['amp'+profile[-1:]] = output.profiles[profile]['amp']
                profile_table['phase'+profile[-1:]] = output.profiles[profile]['phase']

            grp.create_dataset('PROFILE_DATA',compression=config.compression,data=profile_table)


        if output.sweep:
            sweep_dtypes = [('sweep_type',np.int),
                            ('sweep_low',np.float),
                            ('sweep_high',np.float),
                            ('sweep_risetime',np.float),
                            ('sweep_falltime',np.float),
                            ('sweep_dt',np.float)]

            sweep_table = np.empty(1, dtype=sweep_dtypes)

            sweep_types = {'freq': 0,
                           'phase': 1,
                           'amp': 2}

            sweep_table['sweep_type'] = sweep_types[output.sweep_params['type']]
            sweep_table['sweep_low'] = output.sweep_params['low']
            sweep_table['sweep_high'] = output.sweep_params['high']
            sweep_table['sweep_risetime'] = output.sweep_params['risetime']
            sweep_table['sweep_falltime'] = output.sweep_params['falltime']
            sweep_table['sweep_dt'] = output.sweep_params['sweep_dt']

            grp.create_dataset('SWEEP_DATA',compression=config.compression,data=sweep_table)



import time

from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab

@BLACS_tab
class AD9914Tab(DeviceTab):
    def initialise_GUI(self):
        # Capabilities
        self.base_units =    {'freq':'MHz',         'amp':'Arb',   'phase':'Degrees'}
        self.base_min =      {'freq':0.0,           'amp':0,       'phase':0}
        self.base_max =      {'freq':1400.,         'amp':1,       'phase':360}
        self.base_step =     {'freq':1.,            'amp':1/1023., 'phase':1}
        self.base_decimals = {'freq':4,             'amp':4,       'phase':3} # TODO: find out what the phase precision is!
        self.num_DDS = 8

        # Create DDS Output objects
        dds_prop = {}
        for i in range(self.num_DDS):
            dds_prop['profile %d'%i] = {}
            for subchnl in ['freq', 'amp', 'phase']:
                dds_prop['profile %d'%i][subchnl] = {'base_unit':self.base_units[subchnl],
                                                     'min':self.base_min[subchnl],
                                                     'max':self.base_max[subchnl],
                                                     'step':self.base_step[subchnl],
                                                     'decimals':self.base_decimals[subchnl]
                                                    }

        # Create the output objects
        self.create_dds_outputs(dds_prop)

        # Create widgets for output objects
        dds_widgets,ao_widgets,do_widgets = self.auto_create_widgets()
        # and auto place the widgets in the UI
        self.auto_place_widgets(("DDS Profiles",dds_widgets))

        # # Store the COM port to be used   !!!
        # self.com_port = str(self.settings['connection_table'].find_by_name(self.device_name).BLACS_connection)
        
        self.instance = int(self.settings['connection_table'].find_by_name(self.device_name).BLACS_connection)
        
        #
        # Create and set the primary worker
        self.create_worker("main_worker", AD9914Worker,{'instance':self.instance})
        self.primary_worker = "main_worker"

        # Set the capabilities of this device
        self.supports_remote_value_check(False) # !!!
        self.supports_smart_programming(False) # !!!

@BLACS_worker
class AD9914Worker(Worker):

    def init(self):
        global h5py; import labscript_utils.h5_lock, h5py

        self._dll = windll.adiddseval

        self._fFindHardware = self._dll.FindHardware
        self._fGetHardwareHandles = self._dll.GetHardwareHandles
        self._fGetHardwareCount = self._dll.GetHardwareCount
        self._fIsConnected = self._dll.IsConnected
        self._fGetPortConfig = self._dll.GetPortConfig
        self._fSetPortConfig = self._dll.SetPortConfig
        self._fGetPortValue = self._dll.GetPortValue
        self._fSetPortValue = self._dll.SetPortValue
        self._fGetSpiInstruction = self._dll.GetSpiInstruction
        self._fSpiRead = self._dll.SpiRead
        self._fSpiWrite = self._dll.SpiWrite


        self.smart_cache = {'PROFILE_DATA': None,
                            'SWEEP_DATA': None}

        self.handle = None
        self.vid = 0x0456
        self.pid = 0xEE1F
        self.portConfig = [0x7f, 0xff, 0x00, 0xff]  # Configures the USB Cypress chip's ports as inputs or outputs
        self.regLength = 4
        self.usePLL = True
        self.clk = 2.44e9 if self.usePLL else 3.5e9

        self.FindHardware()

        if not self.IsConnected():
            raise LabscriptError("Device is not connected")

        self.SetPortConfig()
        self.Reset()

        if (self.usePLL):
            # *** Use 244x PLL multiplier w/ 10MHz clock input
            reg2 = bytearray("\x1C\x7A\x04\x00")
            self.WriteRegister(2, reg2)

        self.CAL()

        reg0 = bytearray("\x08\x01\x01\x00")
        reg1 = bytearray("\x00\x09\x80\x00")
        self.WriteRegister(0, reg0)
        self.WriteRegister(1, reg1)

        self.IOUpdate()

        # Hack - must always restart device on first startup
        if self.ReadRegister(0) != reg0:
            raise LabscriptError("Registers did not reset correctly - please restart the device")
            

        # for addr in range(18):
        #     print repr(self.ReadRegister(addr))

    def Reset(self):
        self.SetPortValue(0,0x7)
        time.sleep(0.03)
        self.SetPortValue(0,0x3)
        time.sleep(0.3)
#        self.SetPortValue(4,0x7)

        # if ss:
        #     SSauce=self.GetSSload(ss)
        #     for (reg,arry) in SSauce:
        #         self.write(reg,arry,1)


    def CAL(self):
        reg03=self.ReadRegister(0x03)
        reg03[3] |= 0x01
        self.WriteRegister(0x03,reg03,True)
        reg03[3] &= 0xFE
        time.sleep(0.4)
        self.WriteRegister(0x03,reg03,True)
        time.sleep(0.1)

        if (self.usePLL):
            # *** Calibrate 244x PLL multiplier w/ 10MHz clock input
            reg00=self.ReadRegister(0x00)
            reg00[3] &= 0xFE
            self.WriteRegister(0x00,reg00,True)
            time.sleep(0.1)
            reg00[3] |= 0x01
            self.WriteRegister(0x00,reg00,True)
            time.sleep(0.4)

            reg1b=self.ReadRegister(0x1B)
            if (reg1b[3] & 0x01) == 0:
                raise RuntimeError("PLL is not locked. Please restart the device.")

    def FindHardware (self):
        print("Locating hardware...")

        vidArry = c_int*1
        pidArry = c_int*1
        vid = vidArry(self.vid)
        pid = pidArry(self.pid)
        length = c_int(1)
        if not self._fFindHardware(byref(vid),byref(pid),length):
            raise LabscriptError("Could not find hardware")

        handleArray = c_int*self._fGetHardwareCount()
        handle = handleArray(0)
        self._fGetHardwareHandles(byref(handle))
        self.handle = c_int(handle[self.instance])


    def IsConnected (self):
        print("Checking connection...")
        connected = bool(self._fIsConnected(self.handle))
        if connected:
            print("Device is connected")

        return connected

    def GetPortConfig (self, port):
        val = c_byte()
        self._fGetPortConfig(self.handle, port, byref(val))
        return val.value

    # Configures the USB Cypress chip's ports as inputs or outputs
    def SetPortConfig (self):
        print("Setting configuration of ports...")

        for port in range(4):
            self._fSetPortConfig(self.handle, port, self.portConfig[port])

    def GetPortValue (self, port):
        data = c_byte()
        self._fGetPortValue(self.handle, port, byref(data))
        return data.value

    def SetPortValue (self, port, data):
        self._fSetPortValue(self.handle, port, data)

    def ReadRegister (self, addr):
        instr = self.GetSpiInstruction(1, addr)

        instr = c_byte(instr)
        regVals = c_uint32()
        self._fSpiRead(self.handle, byref(instr), sizeof(instr), byref(regVals), self.regLength, 0)

        return bytearray(struct.pack('@I', regVals.value))

    def WriteRegister (self, addr, data, update=False):

        if not isinstance(data, bytearray):
            raise  LabscriptError("data must be a bytearray")
        if len(data) != self.regLength:
            raise LabscriptError("Length of data bytearray does not match register length")

        instr = self.GetSpiInstruction(0, addr)

        dataCopy = bytearray(4)
        dataCopy[:] = data

        dataCopy.append(instr)
        dataCopy.append(0)
        dataCopy.append(0)
        dataCopy.append(0)

        writeData = c_uint64(struct.unpack('Q',dataCopy)[0])
        self._fSpiWrite(self.handle, byref(writeData), self.regLength+1, 0)

        if update:
            self.IOUpdate()


    def GetSpiInstruction (self, rw, addr):
        instr = c_byte()
        self._fGetSpiInstruction(rw, addr, byref(instr), sizeof(instr))
        return instr.value

    def IOUpdate (self):
        # Pushes instructions from the buffer into the actual DDS register (must do this in order for the DDS to actually update its signal)
        
        self.SetPortValue(1, 0xF8)
        time.sleep(0.1)
        self.SetPortValue(1, 0xF0)
        time.sleep(0.1)
        
        
    def CalcFTW (self, fout):   # fout in MHz
        fout = fout * 1e6   # convert from MHz to Hz
        FTWint = round(2**32 * (fout / self.clk))

#        print fout
#        print FTWint

        if FTWint >= 2**32:
            raise LabscriptError("Frequency "+str(fout)+"MHz is too high.")

        return bytearray(struct.pack('Q', FTWint))[:4]

    def CalcFreq (self, FTW):
        if not isinstance(FTW, bytearray):
            raise LabscriptError("FTW must be a bytearray")

        FTWint = float(struct.unpack('L', FTW)[0])

        return (FTWint / 2**32) * self.clk / 1e6    # Convert from Hz to MHz

    def CalcPOW (self, dp):     # dp in degrees
        dp %= 360   # Make sure dp is always in [0,360)

        POWint = round(2**16 * (dp / 360))

        if POWint >= 2**16:
            raise LabscriptError("Phase "+str(dp)+" is too high.")

        return bytearray(struct.pack('L', POWint))[:2]

    def CalcPhase (self, POW):
        if not isinstance(POW, bytearray):
            raise LabscriptError("POW must be a bytearray")

        POWint = float(struct.unpack('H', POW)[0])

        return (POWint / 2**16) * 360

    def CalcASF (self, ampl):     # ampl as a fraction of full amplitude

        if ampl == 1.0:
            ASFint = 2**12 - 1
        else:
            ASFint = round(2**12 * ampl)

        if ASFint >= 2**12:
            raise LabscriptError("Amplitude "+str(ampl)+" is too high.")

        return bytearray(struct.pack('L', ASFint))[:2]

    def CalcAmpl (self, ASF):
        if not isinstance(ASF, bytearray):
            raise LabscriptError("ASF must be a bytearray")

        ASFint = float(struct.unpack('H', ASF)[0])

        return ASFint / 2**12

    def SetUpSweep (self, type, low, high, risetime, falltime, dt):

        # Set profiles to default values
        # !!!!! Currently assumes that:
        #           1. Only frequencies are swept (NOT ampl or phase)
        #           2. Frequency sweeps occur with amplitude = 1 and phase = 0
        for profile in range(8):
            self.program_static('profile %i'%profile, 'freq', 13)   # Arbitrary frequency
            self.program_static('profile %i'%profile, 'amp', 1)
            self.program_static('profile %i'%profile, 'phase', 0)
        
        
        reg0 = self.ReadRegister(0)
        reg0[1] |= 0x10         # Clear accumulator
        self.WriteRegister(0, reg0, True)
        time.sleep(0.1)
        reg0[1] &= ~0x10
        self.WriteRegister(0, reg0, True)

        reg1 = self.ReadRegister(1)
        reg1[2] |= 0x08         # Enable digital ramp
        reg1[2] &= ~0x30        # Clear ramp destination bits
        reg1[2] |= type << 4    # Set ramp destination bits
        self.WriteRegister(1, reg1)

        if low >= high:
            raise LabscriptError("High sweep point must be higher than low sweep point.")

        stepSize_rise = (high - low) * dt / risetime
        stepSize_fall = (high - low) * dt / falltime
        
#        print "Step sizes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#        print stepSize_rise
#        print stepSize_fall

        reg4 = bytearray(4)
        reg5 = bytearray(4)
        reg6 = bytearray(4)
        reg7 = bytearray(4)
        if type == 0:       # freq
            reg4 = self.CalcFTW(low)
            reg5 = self.CalcFTW(high)
            reg6 = self.CalcFTW(stepSize_rise)
            reg7 = self.CalcFTW(stepSize_fall)
        elif type == 1:     # phase
            # Must shift over by 18 bits to fill only the 14 MSBs
            reg4[2:] = bytearray(struct.pack('@H', int(struct.unpack('@H', self.CalcPOW(low))[0]) * (2**2) ))[:2]
            reg5[2:] = bytearray(struct.pack('@H', int(struct.unpack('@H', self.CalcPOW(high))[0]) * (2**2) ))[:2]
            reg6[2:] = bytearray(struct.pack('@H', int(struct.unpack('@H', self.CalcPOW(stepSize_rise))[0]) * (2**2) ))[:2]
            reg7[2:] = bytearray(struct.pack('@H', int(struct.unpack('@H', self.CalcPOW(stepSize_fall))[0]) * (2**2) ))[:2]
        elif type == 2:     # amp
            # Must shift over by 20 bits to fill only the 12 MSBs
            reg4[2:] = bytearray(struct.pack('@H', int(struct.unpack('@H', self.CalcASF(low))[0]) * (2**4) ))[:2]
            reg5[2:] = bytearray(struct.pack('@H', int(struct.unpack('@H', self.CalcASF(high))[0]) * (2**4) ))[:2]
            reg6[2:] = bytearray(struct.pack('@H', int(struct.unpack('@H', self.CalcASF(stepSize_rise))[0]) * (2**4) ))[:2]
            reg7[2:] = bytearray(struct.pack('@H', int(struct.unpack('@H', self.CalcASF(stepSize_fall))[0]) * (2**4) ))[:2]

        if int(struct.unpack('@I', reg6)[0]) == 0:
            raise LabscriptError("Rising step size is too small - got rounded to 0")
        if int(struct.unpack('@I', reg7)[0]) == 0:
            raise LabscriptError("Falling step size is too small - got rounded to 0")

        self.WriteRegister(4, reg4)
        self.WriteRegister(5, reg5)
        self.WriteRegister(6, reg6)
        self.WriteRegister(7, reg7)

        # Set the slope time interval (constant: 2e-6 s)
        dt_reg = bytearray(4)
        dt_word = round(dt * self.clk / 24)
        if dt_word >= 2**16:
            raise LabscriptError("Sweep time interval dt is too long.")

        dt_word = bytearray(struct.pack('@H', dt_word))
        dt_reg[:2] = dt_word
        dt_reg[2:] = dt_word
        self.WriteRegister(8, dt_reg)

        self.IOUpdate()
        
    def DumpRegisters(self, filename=None):
        if filename:
            text_file = open(filename, "w")
            for r in range(0,28):
                text_file.write('Register ' + str(r) + ':\t' + binascii.hexlify(self.ReadRegister(r)) + '\n')
            for r in range(0,4):
                text_file.write('Port ' + str(r) + ':\t' + str(self.GetPortValue(r)) + '\n')
            text_file.close()
        else:
            for r in range(0,28):
                print 'Register ' + str(r) + ':\t' + binascii.hexlify(self.ReadRegister(r))
            for r in range(0,4):
                print 'Port ' + str(r) + ':\t' + str(self.GetPortValue(r))

    def check_remote_values(self):

        results = {}

        for profile in range(8):
            results['profile %d'%profile] = {}

            freqData = self.ReadRegister(11 + 2 * profile)
            results['profile %d'%profile]['freq'] = self.CalcFreq(freqData)

            phAmplData = self.ReadRegister(12 + 2 * profile)
            results['profile %d'%profile]['phase'] = self.CalcPhase(phAmplData[:2])
            results['profile %d'%profile]['amp'] = self.CalcAmpl(phAmplData[2:])
        return results


    def program_manual(self,front_panel_values):
        for profile in front_panel_values:
            for subchnl in ['freq','amp','phase']:
                self.program_static(profile,subchnl,front_panel_values[profile][subchnl])

        self.IOUpdate()
        return self.check_remote_values()

    def program_static(self,profile,type,value):

        profile = int(profile[-1:])

        if type == 'freq':
            addr = 11 + 2 * profile
            self.WriteRegister(addr, self.CalcFTW(value))

        elif type == 'phase':
            addr = 12 + 2 * profile

            regVal = self.ReadRegister(addr)
            phaseVal = self.CalcPOW(value)
            regVal[0] = phaseVal[0]
            regVal[1] = phaseVal[1]

            self.WriteRegister(addr, regVal)

        elif type == 'amp':
            addr = 12 + 2 * profile

            regVal = self.ReadRegister(addr)
            ampVal = self.CalcASF(value)
            regVal[2] = ampVal[0]
            regVal[3] = ampVal[1]

            self.WriteRegister(addr, regVal)
        else:
            raise LabscriptError("Unrecognized DDS sub-profile type")

        # Now that a static update has been done, we'd better invalidate the saved PROFILE_DATA:
        self.smart_cache['PROFILE_DATA'] = None

    def transition_to_buffered(self,device_name,h5file,initial_values,fresh):

        # Store the initial values in case we have to abort and restore them:
        self.initial_values = initial_values

        # Store the final values to for use during transition_to_static:
        self.final_values = {}
        profile_data = None
        sweep_data = None

        with h5py.File(h5file) as hdf5_file:
            group = hdf5_file['/devices/'+device_name]
            # If there are values to set the unbuffered outputs to, set them now:
            if 'PROFILE_DATA' in group:
                profile_data = group['PROFILE_DATA'][:][0]

            if 'SWEEP_DATA' in group:
                sweep_data = group['SWEEP_DATA'][:][0]

        if (profile_data is not None) and (sweep_data is not None):
            raise LabscriptError("AD9914 does not support Profile mode and Sweep mode simultaneously")

        if profile_data is not None:
            reg1 = self.ReadRegister(1)
            reg1[2] &= ~0x08         # Disable digital ramp
            self.WriteRegister(1, reg1)
        
            data = profile_data
            if fresh or data != self.smart_cache['PROFILE_DATA']:
#                self.logger.debug('Static data has changed, reprogramming.')
                self.smart_cache['PROFILE_DATA'] = data

                for name in data.dtype.names:
                    profile = int(name[-1:])
                    type = name[:-1]
                    self.program_static(name, type, data[name])

                    # Save these values into final_values so the GUI can
                    # be updated at the end of the run to reflect them:
                    profileKey = 'profile %d'%profile
                    if not (profileKey) in self.final_values:
                        self.final_values[profileKey] = {}
                    self.final_values[profileKey][type] = data[name]

            # Update the DDS board outputs
            self.IOUpdate()

        if sweep_data is not None:
            self.SetUpSweep(sweep_data['sweep_type'], sweep_data['sweep_low'], sweep_data['sweep_high'], sweep_data['sweep_risetime'], sweep_data['sweep_falltime'], sweep_data['sweep_dt'])
    
        return self.final_values

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def abort_buffered(self):
        return self.transition_to_manual(True)

    def transition_to_manual(self,abort = False):
        
        if abort:
            # If we're aborting the run, then we need to reset the Static DDSs to their initial values.
            # We also need to invalidate the smart programming cache.
            values = self.initial_values
            self.smart_cache['PROFILE_DATA'] = None
        else:
            # If we're not aborting the run, then we need to set the Static DDSs to their final values.
            values = self.final_values

        for profile in values:
            for subchnl in ['freq','amp','phase']:
                self.program_static(profile,subchnl,values[profile][subchnl])

        return True

    def shutdown(self):
        return



