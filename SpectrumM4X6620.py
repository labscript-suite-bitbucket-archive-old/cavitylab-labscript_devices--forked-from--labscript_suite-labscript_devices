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
from scipy.fftpack import fft, rfft, irfft, ifft
from scipy.signal import chirp
import labscript_utils.h5_lock, h5py
from ctypes import *
import struct

##### All the data for the sequence is wrapped up in these class structures. #####
class pulse():
    def __init__(self,start_freq,end_freq,phase,amplitude,ramp_type):
        self.start = start_freq
        self.end = end_freq
        self.phase = phase
        self.amp = amplitude
        self.ramp_type = ramp_type ## String. Can be linear, quadratic, None

class segment():
    def __init__(self,t,duration,loops=1):
        self.time = t
        self.duration = duration
        self.pulses = []
        self.loops = loops

    def add_pulse(self,start_freq,end_freq,phase,amplitude,ramp_type):
        self.pulses.append(pulse(start_freq,end_freq,phase,amplitude,ramp_type))

# Structure which contains a list of segments (frequency comb, ramps, single frequencies, etc.)
class channel():
    def __init__(self,name='',power=0,port=0):
        self.segments = []
        self.power = power
        self.device_name = name
        self.port = port

    def add_segment(self, segment):
        self.segments.append(segment)

class channel_data():
    def __init__(self,channel_names,powers,mode,clock_freq):
        self.channels = []
        self.mode = mode
        self.clock_freq = clock_freq
        for i in range(len(channel_names)):
            if channel_names[i] is not '':
                ch = channel(channel_names[i],powers[i],port=i)
                self.channels.append(ch)

@labscript_device
class SpectrumM4X6620(IntermediateDevice):

    def __init__(self,name,parent_device):
        IntermediateDevice.__init__(self,name,parent_device)
        self.BLACS_connection = 5
        self.set_mode('Off') ## Initialize data structure

    ## Sets up the channel_data structure that will be filled with the following function calls (single_freq,comb,sweep...).
    def set_mode(self,mode_name, clock_freq=500, channels = ['','','',''], powers = [0,0,0,0]):
        self.channel_data = channel_data(channel_names=channels,powers=powers,mode=mode_name,clock_freq=clock_freq)

    def reset(self, t): # This doesn't do anything but must be here.
        return t

    def single_freq(self, t, duration, freq, amplitude, phase, ch, loops = 1):
        self.sweep_comb(t,duration,[freq],[freq],[amplitude],[phase],ch,'None',loops)

    def sweep(self, t, duration, start_freq, end_freq, amplitude, phase, ch, ramp_type):
        self.sweep_comb(t, duration, [start_freq], [end_freq], [amplitude], [phase], ch, ramp_type)

    def comb(self,t,duration,freqs,amplitudes,phases,ch,loops=1):
        self.sweep_comb(t,duration,freqs,freqs,amplitudes,phases,ch,'None',loops)

    # Function that allows user to initialize a segment.
    def sweep_comb(self, t, duration, start_freqs, end_freqs, amplitudes, phases, ch, ramp_type,loops=1):
        seg = segment(t,duration,loops)
        for i in range(len(start_freqs)):
            if (amplitudes[i] < 0) or (amplitudes[i] > 1):
                raise LabscriptError("Amplitude[" + str(i) + "] = " + str(amplitudes[i]) + " is outside the allowed range [0,1]")
            seg.add_pulse(start_freqs[i],end_freqs[i],phases[i],amplitudes[i],ramp_type)
        self.channel_data.channels[ch].add_segment(seg)
        return t+duration

    # Function specifically used for tweezers.
    def tweezers(self,t,number,loop=True):
        if loop: loops = 0
        else: loops = 1

        freqs = np.linspace(start=MEGA(85),stop=MEGA(120),num=number)
        phases=np.random.rand(number)
        amplitudes = [2000 for i in range(number)]
        self.comb(t,.002,freqs,amplitudes,phases,0,loops)


    # Load profile table containing data into h5 file, using the same hierarchical structure from above.
    def generate_code(self, hdf5_file):
        device = hdf5_file.create_group('/devices/' + self.name)

        #Store device settings
        profile_dtypes = np.dtype([('mode', 'S10'),('clock_freq',np.float)])
        profile_table = np.array((0,0),dtype=profile_dtypes)
        profile_table['mode'] = self.channel_data.mode
        profile_table['clock_freq'] = self.channel_data.clock_freq
        device.create_dataset('device_settings', data=profile_table)

        # Store channel settings
        for j in range(len(self.channel_data.channels)):
            channel = self.channel_data.channels[j]
            c = device.create_group(channel.device_name)
            profile_dtypes = [('power', np.float),('device_name','S10'),('port',int)]
            profile_table = np.array((0,0,0), dtype=profile_dtypes)
            profile_table['power'] = channel.power
            profile_table['device_name'] = channel.device_name
            profile_table['port'] = channel.port
            c.create_dataset('channel_settings', data=profile_table)

            # Store segment settings
            for segment in channel.segments:
                name = "Segment: t = " + str(segment.time) + ", dur = " + str(segment.duration)
                if name in c:   ## If segment already exists, add to already created group
                    grp = c[name]
                else:
                    grp = c.create_group(name)
                    profile_dtypes = [('start_time', np.float), ('duration', np.float),('loops',int)]
                    profile_table = np.zeros(1, dtype=profile_dtypes)
                    profile_table['start_time'] = segment.time
                    profile_table['duration'] = segment.duration
                    profile_table['loops'] = segment.loops
                    grp.create_dataset('segment_settings', data=profile_table)

                # Store pulses
                profile_dtypes = [('start_freq', np.float),
                                  ('end_freq', np.float),
                                  ('phase', np.float),
                                  ('amp', np.float),
                                  ('ramp_type',"S10")]
                profile_table = np.zeros(len(segment.pulses), dtype=profile_dtypes)
                for i in range(len(segment.pulses)):
                    pulse = segment.pulses[i]

                    profile_table['start_freq'][i] = pulse.start
                    profile_table['end_freq'][i] = pulse.end
                    profile_table['phase'][i] = pulse.phase
                    profile_table['amp'][i] = pulse.amp
                    profile_table['ramp_type'][i] = pulse.ramp_type

                if 'pulse_data' in grp: ### If segment already has associated data, add to the existing dataset.
                    d = grp['pulse_data']
                    d.resize((d.shape[0]+profile_table.shape[0]), axis=0)
                    d[-profile_table.shape[0]:] = profile_table
                else:
                    grp.create_dataset('pulse_data', maxshape=(1000,),
                                       data=profile_table, dtype = profile_dtypes, chunks = True)
@BLACS_tab
class SpectrumM4X6620Tab(DeviceTab):
    def initialise_GUI(self):
        # Capabilities
        self.base_units =    {'freq':'MHz',         'Power':'dBm',   'phase':'Degrees'}
        self.base_min =      {'freq':0.1,           'Power':-136.0,  'phase':0}
        self.base_max =      {'freq':4000.,         'Power':25.0,    'phase':360}
        self.base_step =     {'freq':1.0,           'Power':1.0,     'phase':1}
        self.base_decimals = {'freq':4,             'Power':4,       'phase':3}

        # Create DDS Output objects
        RF_prop = {}
        RF_prop['channel 0'] = {}
        for subchnl in ['freq', 'Power', 'phase']:
            RF_prop['channel 0'][subchnl] = {'base_unit':self.base_units[subchnl],
                                                'min':self.base_min[subchnl],
                                                'max':self.base_max[subchnl],
                                                'step':self.base_step[subchnl],
                                                'decimals':self.base_decimals[subchnl]
                                                }

        # Create the output objects
        self.create_dds_outputs(RF_prop)

        # Create widgets for output objects
        dds_widgets,ao_widgets,do_widgets = self.auto_create_widgets()
        # and auto place the widgets in the UI
        self.auto_place_widgets(("RF Output",dds_widgets))

        # Create and set the primary worker
        self.create_worker("main_worker", SpectrumM4X6620Worker)
        self.primary_worker = "main_worker"

        # Set the capabilities of this device
        self.supports_remote_value_check(False)
        self.supports_smart_programming(False)


@BLACS_worker
class SpectrumM4X6620Worker(Worker):
    def init(self):
        self.final_values = {'channel 0' : {}}
        self.remote = False
        if self.remote:
            self.card = spcm_hOpen("TCPIP::171.64.58.206::INSTR")  ### remote card mode only for testing purposes
        else:
            self.card = spcm_hOpen(create_string_buffer (b'/dev/spcm0'))
        if self.card == None:
            raise LabscriptError("Device is not connected.")

        self.power = 0
        self.bytesPerSample = 2

        # self.samples = 1000
        # self.buffer = create_string_buffer(self.samples)


    def card_settings(self):
        ## General settings -- mode specific settings are defined in transition_to_buffered
        err=spcm_dwSetParam_i32(self.card, SPC_CLOCKMODE, SPC_CM_INTPLL)  # clock mode internal PLL
        if err: raise LabscriptError("Error detected in settings: " + str(err))
        err=spcm_dwSetParam_i32(self.card, SPC_SAMPLERATE, int32(self.clock_freq))
        if err: raise LabscriptError("Error detected in settings: " + str(err))
        err=spcm_dwSetParam_i32(self.card, SPC_TRIG_EXT1_MODE, SPC_TM_POS)
        if err: raise LabscriptError("Error detected in settings: " + str(err))  ###### ERROR: CARD STILL RUNNING, WHEN USING BLACS REPEAT
        err=spcm_dwSetParam_i32(self.card, SPC_TRIG_ORMASK, SPC_TMASK_EXT1)
        if err: raise LabscriptError("Error detected in settings: " + str(err))

        if self.mode == 'multi':
            spcm_dwSetParam_i32(self.card, SPC_CARDMODE, SPC_REP_STD_MULTI)
            if err: raise LabscriptError("Error detected in settings: " + str(err))
        elif self.mode == 'single':
            err = spcm_dwSetParam_i32(self.card, SPC_CARDMODE, SPC_REP_STD_SINGLE)
            if err: raise LabscriptError("Error detected in settings: " + str(err))
        elif self.mode == 'sequence':
            raise LabscriptError("Not a valid mode yet.")
        else: self.mode == 'Off' # Default


### These functions don't currently do anything useful, but they could be used to disply useful info on GUI ###
    def check_remote_values(self):
        results = {'channel 0': {}}
        results['channel 0']['freq'] = 0
        results['channel 0']['Power'] = self.power
        results['channel 0']['phase'] = 0
        return results
    def program_manual(self,front_panel_values):
        return self.check_remote_values()


    ## Uses class structure to generate the necessary buffer to be sent to the Spectrum Card.
    ## How the buffer is organized is dependent on the mode being used. In single mode, there is a single segment,
    ## but with the possibility of looping over that segment. In multimode, segments are added in a row, with zero
    ## padding where necessary.
    ## The sequence mode has not been implemented yet.
    ## Multichannel is not fully functional yet. For now, just the first channel is played.
    ## To handle exceptions, the function returns False if a buffer was not generated, so as not to send this
    ## information to the card. Otherwise, the function returns True.
    def generate_buffer(self):

        # Iterate over the channels which are on. Set channel-specific
        for channel in self.channels:

            ##### Check power setting -- there are hardcoded limits on power for specific devices #####
            if channel.device_name == 'microwaves':
                n=0 ## ADD LIMIT FOR THIS DEVICE
            elif channel.device_name == 'tweezers':
                if channel.power > 5: raise LabscriptError("POWER SETTING TOO HIGH. The power should be a maximum of 5 dBm for this device.")
            else:
                raise LabscriptError("Device name not recognized. The only available devices are 'tweezers' and 'microwaves'.")

            ## Setting amplitude corresponding to chosen power.
            amplitude = int(np.sqrt(0.1) * 10 ** (float(channel.power) / 20.0) * 1000)
            if amplitude < 80: raise LabscriptError("Power below acceptable range. Min power = -23.9 dBm")
            if amplitude > 2500: raise LabscriptError("Power above acceptable range. Max power = 5.9 dBm.")
            if channel.port == 0:
                err=spcm_dwSetParam_i32(self.card, SPC_AMP0, int32(amplitude))
                if err: raise LabscriptError("Error detected in settings: " + str(err) + "Amplitude: " + str(self.amplitude))
                err = spcm_dwSetParam_i32(self.card, SPC_CHENABLE, CHANNEL0)
                if err: raise LabscriptError("Error detected in settings: " + str(err))
                err = spcm_dwSetParam_i32(self.card, SPC_ENABLEOUT0, 1)
                if err: raise LabscriptError("Error detected in settings: " + str(err))
            if channel.port == 1:
                err=spcm_dwSetParam_i32(self.card, SPC_AMP1, int32(amplitude))
                if err: raise LabscriptError("Error detected in settings: " + str(err) + "Amplitude: " + str(self.amplitude))
                err = spcm_dwSetParam_i32(self.card, SPC_CHENABLE, CHANNEL1)
                if err: raise LabscriptError("Error detected in settings: " + str(err))
                err = spcm_dwSetParam_i32(self.card, SPC_ENABLEOUT1, 1)
                if err: raise LabscriptError("Error detected in settings: " + str(err))
            if channel.port == 2:
                err=spcm_dwSetParam_i32(self.card, SPC_AMP2, int32(amplitude))
                if err: raise LabscriptError("Error detected in settings: " + str(err) + "Amplitude: " + str(self.amplitude))
                err = spcm_dwSetParam_i32(self.card, SPC_CHENABLE, CHANNEL2)
                if err: raise LabscriptError("Error detected in settings: " + str(err))
                err = spcm_dwSetParam_i32(self.card, SPC_ENABLEOUT2, 1)
                if err: raise LabscriptError("Error detected in settings: " + str(err))
            if channel.port == 3:
                err=spcm_dwSetParam_i32(self.card, SPC_AMP3, int32(amplitude))
                if err: raise LabscriptError("Error detected in settings: " + str(err) + "Amplitude: " + str(self.amplitude))
                err = spcm_dwSetParam_i32(self.card, SPC_CHENABLE, CHANNEL3)
                if err: raise LabscriptError("Error detected in settings: " + str(err))
                err = spcm_dwSetParam_i32(self.card, SPC_ENABLEOUT3, 1)
                if err: raise LabscriptError("Error detected in settings: " + str(err))

            #### MULTI MODE #### Each segment must be same size.
            elif (self.mode == 'multi'):
#                 lSetChannels = int32 (0)
#                 spcm_dwGetParam_i32 (self.card, SPC_CHCOUNT,     byref (lSetChannels))
#                 print(lSetChannels.value)
#                 lBytesPerSample = int32 (0)
#                 spcm_dwGetParam_i32 (self.card, SPC_MIINST_BYTESPERSAMPLE,  byref (lBytesPerSample))
#                 print(lBytesPerSample.value)

                ch = self.channels[0]
                if not ch.segments: return False

                self.num_segments = len(ch.segments)
                max_samples = 0 ### For multimode, we must know the largest size segment, in order to make each segment the same size.
                for seg in ch.segments:
                    num = seg.duration * self.clock_freq
                    if num > max_samples: max_samples = num
                self.samples = int(max_samples)
                if (self.samples % 32) != 0: raise LabscriptError("Number of samples must be a multiple of 32") # Not *strictly* necessary: see p.105 of Spectrum manual
                size = uint64(self.num_segments * self.samples * self.bytesPerSample)
                self.buffer = create_string_buffer(size.value)
                waveform = cast(self.buffer, ptr16)
                for j in range(self.num_segments):
                    seg = ch.segments[j]
                    samp = self.samples
                    t = np.linspace(0, seg.duration, samp)
                    for pulse in seg.pulses:
                        if pulse.ramp_type == "linear":
                            ramp = pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=seg.duration, f1=pulse.end, method='linear', phi=pulse.phase)
                            for i in range(samp):
                                waveform[j*samp + i] = np.int16(ramp[i])
                        elif pulse.ramp_type == "quadratic":
                            ramp = pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=seg.duration, f1=pulse.end, method='quadratic', phi=pulse.phase)
                            for i in range(samp):
                                waveform[j*samp + i] = np.int16(ramp[i])
                        else: ## If no allowed ramp is specified, then it is assumed that the frequency remains stationary.
                            for i in range(samp):
                                waveform[j*samp + i] = np.int16(pulse.amp * (2**15-1) * math.sin(2 * np.pi * pulse.start * i / self.clock_freq))

#                 with open('X:\\dataDump.csv', 'w') as f:
#                     for i in range(0, self.samples * self.num_segments):
#                         print >> f, waveform[i]
                ## Card settings specific to multimode
                spcm_dwSetParam_i32(self.card, SPC_MEMSIZE, self.samples * self.num_segments)     # !!! Must extend to account for other channels?
                spcm_dwSetParam_i32(self.card, SPC_SEGMENTSIZE, self.samples)

            #### SEQUENCE MODE ####
            elif (self.mode == 'sequence'):
                raise LabscriptError("Mode not currently available.")

            #### SINGLE MODE #### -- useful to loop over one frequency for an extended period of time.
            elif (self.mode == 'single'):
                ch = self.channels[0]
                if not ch.segments: return False
                seg = ch.segments[0]
                self.num_segments = 1
                self.loops = seg.loops
                self.samples = int(seg.duration*self.clock_freq)
                if (self.samples % 32) != 0: raise LabscriptError("Number of samples must be a multiple of 32") # Not *strictly* necessary: see p.105 of Spectrum manual
                size = uint64(self.samples * self.bytesPerSample)
                self.buffer = create_string_buffer(size.value)
                waveform = cast(self.buffer, ptr16)
                t = np.linspace(0,seg.duration,self.samples)
                for pulse in seg.pulses:
                    if pulse.ramp_type == "linear":
                        ramp = pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=seg.duration, f1=pulse.end, method='linear', phi=pulse.phase)
                        for i in range(self.samples):
                            waveform[i] = np.int16(ramp[i])
                    elif pulse.ramp_type == "quadratic":
                        ramp = pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=seg.duration, f1=pulse.end, method='quadratic', phi=pulse.phase)
                        for i in range(self.samples):
                            waveform[i] = np.int16(ramp[i])
                    else:
                        for i in range(0,self.samples,1):
                            waveform[i] = np.int16(pulse.amp * (2**15-1) * math.sin(2 * np.pi * pulse.start * i / self.clock_freq + pulse.phase))


                #### RESET AMPLITUDE BASED ON NUMBER OF PULSES ####
                # new_amplitude = amplitude / 3
                # err=spcm_dwSetParam_i32(self.card, SPC_AMP2, int32(new_amplitude))
                # if err: raise LabscriptError("Error detected in settings: " + str(err))

                err = spcm_dwSetParam_i32(self.card, SPC_MEMSIZE, self.samples)
                if err: raise LabscriptError("Error detected in settings: " + str(err))
                err = spcm_dwSetParam_i32(self.card, SPC_LOOPS, uint32(self.loops))
                if err:
                    raise LabscriptError("Error detected in settings: " + str(err))
        return True

    def set_trigger(self):
        if self.mode == 'Off': return

        error = spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER) # | M2CMD_CARD_WAITTRIGGER)
        if error != 0: raise LabscriptError("Error detected during data transfer to card. Error: " + str(error))
        # dwError = spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_WAITREADY)

    def transfer_buffer(self):
        if self.mode == 'Off': return
        dw = spcm_dwDefTransfer_i64 (self.card, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32 (0), self.buffer, uint64 (0), uint64(self.samples * self.num_segments * self.bytesPerSample))
        dwError = spcm_dwSetParam_i32 (self.card, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
        if((dw + dwError) != 0): raise LabscriptError("Error detected during data transfer to card. Error: " + str(dw) + " " + str(dwError))
    class channel():
        def __init__(self, name, power, port):
            self.segments = []
            self.power = power
            self.device_name = name
            self.port = port

        def add_segment(self, segment):
            self.segments.append(segment)


    def transition_to_buffered(self,device_name,h5file,initial_values,fresh):
        # return self.final_values
        self.channels = []
        with h5py.File(h5file) as file:
            device = file['/devices/' +device_name]
            for chan in device.iterkeys():
                if chan == 'device_settings':
                    settings = device[chan]
                    self.mode = settings['mode']
                    self.clock_freq = MEGA(int(settings['clock_freq']))
                else:
                    c = device[chan]
                    ch = channel()
                    for segmt in c.iterkeys():
                        if segmt == 'channel_settings':
                            ch.power = c['channel_settings']['power']
                            ch.device_name = c['channel_settings']['device_name']
                            ch.port = c['channel_settings']['port']
                        else:
                            seg = segment(0, 0)
                            s = c[segmt]
                            for p in s.iterkeys():
                                if p == 'segment_settings':
                                    seg.time = s['segment_settings']['start_time'][0]
                                    seg.duration = s['segment_settings']['duration'][0]
                                    seg.loops = s['segment_settings']['loops'][0]
                                if p == 'pulse_data':
                                    dset = s['pulse_data']
                                    ###### Figure out how to go through every line of the profile table.
                                    for i in range(dset.shape[0]):
                                        start_freq = dset['start_freq'][i]
                                        end_freq = dset['end_freq'][i]
                                        phase = dset['phase'][i]
                                        amplitude = dset['amp'][i]
                                        ramp_type = dset['ramp_type'][i]
                                        seg.add_pulse(start_freq, end_freq, phase, amplitude,ramp_type)
                            ch.segments.append(seg)
                    self.channels.append(ch) ## structure that now holds all necessary data from h5 file

        buf_gen = self.generate_buffer()
        if buf_gen is True:
            ### Card Settings ###
            self.card_settings()

            ### Send Buffer and Set Trigger ###
            self.transfer_buffer()
            self.set_trigger()

        return self.final_values ### NOT CURRENTLY USED FOR ANY INFO TO SEND TO GUI. NECESSARY TO WORK WITH BLACS.

    ### Other Functions, manual mode is not utilized for the spectrum instrumentation card. ###
    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)
    def abort_buffered(self):
        return self.transition_to_manual(True)
    def transition_to_manual(self,abort = False):
        if abort:
            self.shutdown()
            self.init()
        else: spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_STOP)
        return True
    def shutdown(self):
        spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_STOP)
        spcm_vClose(self.card)
