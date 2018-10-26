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
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import gc

##### All the data for the sequence is wrapped up in these class structures. #####
class pulse():
    def __init__(self,start_freq,end_freq,phase,amplitude,ramp_type):
        self.start = start_freq
        self.end = end_freq
        self.phase = phase
        self.amp = amplitude
        self.ramp_type = ramp_type ## String. Can be linear, quadratic, None

class waveform():
    def __init__(self,time,duration,port,loops=1,is_periodic=False,pulses=[]):
        self.time = time
        self.duration = duration
        self.port = port     # !!! Or use channel name?
        self.loops = loops

        self.is_periodic = is_periodic

        # Make new copies of pulses
        self.pulses = []
        for p in pulses:
            self.pulses.append(pulse(p.start,p.end,p.phase,p.amp,p.ramp_type))

        self.sample_start = 0
        self.sample_end = duration

    def add_pulse(self,start_freq,end_freq,phase,amplitude,ramp_type):
        self.pulses.append(pulse(start_freq,end_freq,phase,amplitude,ramp_type))

# Structure which contains a list of waveforms (frequency comb, ramps, single frequencies, etc.)
class waveform_group():
    def __init__(self,time,duration,waveforms,loops=1):
        self.time = time
        self.duration = duration
        self.waveforms = waveforms
        self.loops = loops

    def add_waveform(self,waveform):
        self.waveforms.append(waveform)

class channel_settings():
    def __init__(self,name='',power=0,port=0):
        self.power = power
        self.name = name
        self.port = port

class sample_data():
    def __init__(self,channels,mode,clock_freq):
        self.waveform_groups = []
        self.mode = mode
        self.clock_freq = clock_freq
        self.channels = channels


@labscript_device
class SpectrumM4X6620(IntermediateDevice):

    def __init__(self,name,parent_device,trigger,triggerDur=10e-6):
        IntermediateDevice.__init__(self,name,parent_device)
        self.BLACS_connection = 5
        self.set_mode('Off') ## Initialize data structure

        self.triggerDur = triggerDur

        self.raw_waveforms = []

        self.cur_id = 0

        if trigger:
            if 'device' in trigger and 'connection' in trigger:
                self.triggerDO = DigitalOut(self.name+'_Trigger', trigger['device'], trigger['connection'])
            else:
                raise LabscriptError('You must specify the "device" and "connection" for the trigger input to the SpectrumM4X6620')
        else:
            raise LabscriptError('No trigger specified for device ' + self.name)

    ## Sets up the channel_data structure that will be filled with the following function calls (single_freq,comb,sweep...).
    def set_mode(self,mode_name, channels=[], clock_freq=500):
        if (len(channels) == 3 or len(channels) > 4):
            raise LabscriptError('SpectrumM4X6620 only supports 1, 2, or 4 channels. Please remove a channel or add a dummy channel')

        channel_objects = []
        for i,channel in enumerate(channels):
            if not (channel['name'] == '' or channel['name'] == None):
                channel_objects.append(channel_settings(channel['name'],channel['power'],i))

        self.sample_data = sample_data(channels=channel_objects,mode=mode_name,clock_freq=clock_freq)

    def reset(self, t): # This doesn't do anything but must be here.
        return t

    def single_freq(self, t, duration, freq, amplitude, phase, ch, loops=1):
        self.sweep_comb(t, duration, [freq], [freq], [amplitude], [phase], ch, 'None', loops)

    def sweep(self, t, duration, start_freq, end_freq, amplitude, phase, ch, ramp_type, loops=1):
        self.sweep_comb(t, duration, [start_freq], [end_freq], [amplitude], [phase], ch, ramp_type, loops)

    def comb(self,t,duration,freqs,amplitudes,phases,ch,loops=1):
        self.sweep_comb(t,duration,freqs,freqs,amplitudes,phases,ch,'None',loops)

    # Function that allows user to initialize a waveform.
    def sweep_comb(self, t, duration, start_freqs, end_freqs, amplitudes, phases, ch, ramp_type, loops=1):

        # Convert from time in seconds to time in sample chunks (1 sample chunk = 32 samples)
        t_c = int(math.floor(float(t * self.sample_data.clock_freq * 10**6) / 32.0))
        duration_c = int(math.floor(float(duration * self.sample_data.clock_freq * 10**6) / 32.0))

        if loops > 2**32 - 1:
            raise LabscriptError('Too many loops requested. Number of loops must be less than 2^32')
        if duration_c < 12:
            raise LabscriptError('Duration of segment is too short. Segment duration must be at least 12 sample chunks ( = 768ns)')

        wvf = waveform(t_c,duration_c,ch,loops,is_periodic=(loops > 1))

        for i in range(len(start_freqs)):
            if (amplitudes[i] < 0) or (amplitudes[i] > 1):
                raise LabscriptError("Amplitude[" + str(i) + "] = " + str(amplitudes[i]) + " is outside the allowed range [0,1]")

            wvf.add_pulse(start_freqs[i],end_freqs[i],phases[i],amplitudes[i],ramp_type)
            self.raw_waveforms.append(wvf)

        return t+(loops*duration)

    # # Function specifically used for tweezers.
    # def tweezers(self,t,number,loop=True):
    #     if loop: loops = 0
    #     else: loops = 1
    #
    #     freqs = np.linspace(start=MEGA(85),stop=MEGA(120),num=number)
    #     phases=np.random.rand(number)
    #     amplitudes = [2000 for i in range(number)]
    #     self.comb(t,.002,freqs,amplitudes,phases,0,loops)


    # Load profile table containing data into h5 file, using the same hierarchical structure from above.
    def generate_code(self, hdf5_file):
        print('---------------------------')
        for waveform in self.raw_waveforms:
            print(waveform)

        print('---------------------------')
        for group in self.sample_data.waveform_groups:
            print(group)

        device = hdf5_file.create_group('/devices/' + self.name)

        # Store device settings
        settings_dtypes = np.dtype([('mode', 'S10'),('clock_freq',np.float)])
        settings_table = np.array((0,0),dtype=settings_dtypes)
        settings_table['mode'] = self.sample_data.mode
        settings_table['clock_freq'] = self.sample_data.clock_freq
        device.create_dataset('device_settings', data=settings_table)

        # Store channel settings
        channel_dtypes = [('power', np.float),('name','S10'),('port',int)]
        channel_table = np.zeros(len(self.sample_data.channels), dtype=channel_dtypes)
        for i, channel in enumerate(self.sample_data.channels):
            channel_table[i]['power'] = channel.power
            channel_table[i]['name'] = channel.name
            channel_table[i]['port'] = channel.port
        device.create_dataset('channel_settings', data=channel_table)

        # Store waveform groups
        g = device.create_group('waveform_groups')
        for i,group in enumerate(self.sample_data.waveform_groups):
            group_folder = g.create_group('group ' + str(i))
            settings_dtypes = [('time', np.int), ('duration', np.int), ('loops', np.int)]
            settings_table = np.array((0,0,0),dtype=settings_dtypes)
            settings_table['time'] = group.time
            settings_table['duration'] = group.duration
            settings_table['loops'] = group.loops
            group_folder.create_dataset('group_settings', data=settings_table)


            # Store waveforms
            for waveform in group.waveforms:
                name = "Waveform: t = " + str(waveform.time) + ", dur = " + str(waveform.duration)
                if name in group_folder:   ## If waveform already exists, add to already created group
                    grp = group_folder[name]
                else:
                    grp = group_folder.create_group(name)
                    profile_dtypes = [('time', int), ('duration', int), ('loops', int), ('port', int), ('sample_start', int), ('sample_end', int)]
                    profile_table = np.zeros(1, dtype=profile_dtypes)
                    profile_table['time'] = waveform.time
                    profile_table['duration'] = waveform.duration
                    profile_table['loops'] = waveform.loops
                    profile_table['port'] = waveform.port
                    profile_table['sample_start'] = waveform.sample_start
                    profile_table['sample_end'] = waveform.sample_end
                    grp.create_dataset('waveform_settings', data=profile_table)

                # Store pulses
                profile_dtypes = [('start_freq', np.float),
                                  ('end_freq', np.float),
                                  ('phase', np.float),
                                  ('amp', np.float),
                                  ('ramp_type',"S10")]
                profile_table = np.zeros(len(waveform.pulses), dtype=profile_dtypes)


                if len(waveform.pulses) == 0:
                    raise LabscriptError('Something went wrong in generating Spectrum card data: waveform does not have any pulse data')

                for i in range(len(waveform.pulses)):
                    pulse = waveform.pulses[i]

                    profile_table['start_freq'][i] = pulse.start
                    profile_table['end_freq'][i] = pulse.end
                    profile_table['phase'][i] = pulse.phase
                    profile_table['amp'][i] = pulse.amp
                    profile_table['ramp_type'][i] = pulse.ramp_type

                if 'pulse_data' in grp: ### If waveform already has associated data, add to the existing dataset.
                    d = grp['pulse_data']
                    d.resize((d.shape[0]+profile_table.shape[0]), axis=0)
                    d[-profile_table.shape[0]:] = profile_table
                else:
                    grp.create_dataset('pulse_data', maxshape=(1000,),
                                       data=profile_table, dtype = profile_dtypes, chunks = True)


    def stop(self):
        self.check_channel_collisions(self.raw_waveforms)

        periodicWvfs = filter(lambda k: k.is_periodic == True, self.raw_waveforms)
        nonPeriodicWvfs = filter(lambda k: k.is_periodic == False, self.raw_waveforms)

        nonPeriodicWvfGroups = self.make_waveform_groups(nonPeriodicWvfs)

        self.sample_data.waveform_groups = self.combine_periodic_nonperiodic_groups(periodicWvfs, nonPeriodicWvfGroups)

        # Sort groups in time order (just in case)
        self.sample_data.waveform_groups = sorted(self.sample_data.waveform_groups, key=lambda k: k.time)


        # max_dur = 0
        # for group in self.sample_data.waveform_groups:
        #     if group.duration > max_dur:
        #         max_dur = group.duration

        # # Fire the trigger at the appropriate times, and check that the timing works out
        # t_prev = float('-inf')
        # for group in self.sample_data.sample_groups:
        #     self.triggerDO.go_high(group.time)
        #     self.triggerDO.go_low(group.time+self.triggerDur)
        #
        #     if self.triggerDur >= group.duration:
        #         raise LabscriptError("Trigger duration too long (i.e. longer than the sample group duration; might not be a problem depending on how the card interprets trigger edges)")
        #
        #     if group.time - t_prev < max_dur:
        #         raise LabscriptError("Maximum group length is larger than the separation between groups! This is an edge case that Greg hoped wouldn't crop up but apparently it has (see Cavity Lab wiki page on 9/25/2018).")
        #
        #     t_prev = group.time
        return


    # Organize an array of waveforms into groups of overlapping waveforms
    def make_waveform_groups(self, waveforms):
        print('Make waveforms')
        waveforms = sorted(waveforms, key=lambda k: k.time)     # Crucial to tell apart edge cases where times are identical

        # List of flags marking start and end times of waveform pieces
        # {t,1} marks the beginning of a piece at time t
        # {t,-1} marks the end of a piece at time t
        flagAddRemoveWvf = []
        for waveform in waveforms:
            flagAddRemoveWvf.append({'t': waveform.time, 'flag': 1})
            flagAddRemoveWvf.append({'t': waveform.time + waveform.loops * waveform.duration, 'flag': -1})

        flagAddRemoveWvf = sorted(flagAddRemoveWvf, key=lambda k: k['t'])

        # Find the times at which groups of pieces begin and end
        # Groups start when the sum of flags at time t changes from 0 to more than 0
        # Groups end when the sum of flags at time t hits 0
        numOverlaps = 0
        groupStartIndices = []
        groupEndIndices = []
        for i in range(0,len(flagAddRemoveWvf)):
            nextNumOverlaps = numOverlaps + flagAddRemoveWvf[i]['flag']

            if (numOverlaps < 0) or (nextNumOverlaps < 0):
                raise LabscriptError("Something went wrong in make_waveform_groups(): numOverlaps should never be negative")

            if numOverlaps == 0:
                groupStartIndices.append(i)

            if nextNumOverlaps == 0:
                groupEndIndices.append(i)

            numOverlaps = nextNumOverlaps

        if len(groupStartIndices) != len(groupEndIndices):
            raise LabscriptError("Something went wrong in make_waveform_groups(): length of groupStartIndices should be equal to length of groupEndIndices")
            return


        groups = []
        totalWvfs = 0
        for i in range(len(groupStartIndices)):
            t0 = flagAddRemoveWvf[groupStartIndices[i]]['t']
            t1 = flagAddRemoveWvf[groupEndIndices[i]]['t']

            wvfsInGroup = filter(lambda k: (k.time >= t0) and (k.time + k.loops * k.duration <= t1), waveforms)

            totalWvfs += len(wvfsInGroup)

            groups.append(waveform_group(t0,t1-t0,wvfsInGroup))


        if totalWvfs != len(waveforms):
            raise LabscriptError("Something went wrong in make_waveform_groups(): totalWvfs after grouping should be equal to total number of waveforms")
            return

        return groups


    # Check for channel collisions (single channel can't do multiple things at once!)
    def check_channel_collisions(self, waveforms):
        for port in range(len(self.sample_data.channels)):
            wvfsPerPort = filter(lambda k: k.port == port, waveforms)

            groupsPerPort = self.make_waveform_groups(wvfsPerPort)
            for i in range(0,len(groupsPerPort)):
                if len(groupsPerPort[i].waveforms) > 1:
                    raise LabscriptError("Port collision: you've instructed port " + str(port) + " to play two waveforms at once")


    # Extracts a section of a periodic waveform between t = (t_start,t_end)
    # Result comes in (at most) 3 parts: a partial waveform starting at t_start, a number of full loops in the middle, and a partial waveform ending at t_end
    def split_periodic_waveforms(self, waveforms, t_start, t_end):

        result_waveforms = []
        overlappedWvfs = filter(lambda k: (k.time <= t_end) and (k.time + k.loops * k.duration >= t_start), waveforms)

        overlapGroups = self.make_waveform_groups(overlappedWvfs)
        for ogroup in overlapGroups:
            if len(ogroup.waveforms) > 1:
                raise LabscriptError('Cannot deal with overlapped periodic waveforms. Please remove the overlapped waveforms')

            wvf = ogroup.waveforms[0]

            t0 = wvf.time
            t1 = wvf.time + wvf.loops * wvf.duration
            dur = wvf.duration

            t_start_p = max(t0,t_start)
            t_end_p = min(t1,t_end)

            t_n = int(t0 + math.ceil(float(t_start_p - t0) / float(dur)) * dur)
            n_full_loops = int(math.floor(float(t_end_p - t_n) / float(dur)))

            # Full middle loops
            if n_full_loops > 0:
                waveform_full_loops = waveform(t_n,dur,wvf.port,loops=n_full_loops,is_periodic=True,pulses=wvf.pulses)
                result_waveforms.append(waveform_full_loops)

            # Partial start loop
            if t_start_p != t_n:
                dt_start = t_start_p - (t_n - dur)
                waveform_partial_start =  waveform(t_start_p,dur-dt_start,wvf.port,loops=1,is_periodic=False,pulses=wvf.pulses)
                waveform_partial_start.sample_start = dt_start
                waveform_partial_start.sample_end = dur
                result_waveforms.append(waveform_partial_start)

            # Partial end loop
            if t_end_p != (t_n + n_full_loops * dur):
                dt_end = t_end_p - (t_n + n_full_loops * dur)
                waveform_partial_end =  waveform(t_n+(n_full_loops * dur),dt_end,wvf.port,loops=1,is_periodic=False,pulses=wvf.pulses)
                waveform_partial_end.sample_start = 0
                waveform_partial_end.sample_end = dt_end
                result_waveforms.append(waveform_partial_end)


        return result_waveforms


    # If part of a periodic waveform overlaps with a nonperiodic group, then add this section of the waveform to the group
    # Add the rest of the periodic waveform as a looping group
    def combine_periodic_nonperiodic_groups(self, periodicWvfs, nonPeriodicWvfGroups):

        result_groups = []

        # Add a dummy group so we can automatically take care of the final gap
        nonPeriodicWvfGroups.append(None)

        for i,group in enumerate(nonPeriodicWvfGroups):

            # ---------------------------------------------------------------------------
            # Handle gap *before* this group (and the final gap)


            if len(nonPeriodicWvfGroups) == 1:     # No groups (only the dummy)
                t_start = float('-inf')
                t_end = float('inf')

            elif i == 0:            # First gap
                t_start = float('-inf')
                t_end = group.time

            elif group == None:     # Final gap
                prev_group = nonPeriodicWvfGroups[i-1]
                t_start = prev_group.time + prev_group.duration
                t_end = float('inf')

            else:                   # Middle gaps
                prev_group = nonPeriodicWvfGroups[i-1]
                t_start = prev_group.time + prev_group.duration
                t_end = group.time

            if t_start != t_end:     # There is actually a gap between groups

                newWvfs = self.split_periodic_waveforms(periodicWvfs, t_start, t_end)

                if len(newWvfs) > 0:

                    newGroups = []
                    for wvf in newWvfs:
                        # Create new groups for each piece of the split waveform
                        newGrp = waveform_group(wvf.time,wvf.duration,[wvf],loops=wvf.loops)
                        newGroups.append(newGrp)

                        wvf.loops = 1
                        wvf.is_periodic = False

                    result_groups.extend(newGroups)

            # ---------------------------------------------------------------------------



            # ---------------------------------------------------------------------------
            # Handle the group itself

            if group != None:
                t_start = group.time
                t_end = group.time + group.duration

                newWvfs = self.split_periodic_waveforms(periodicWvfs, t_start, t_end)

                # Add the pieces of the split waveform to the group
                for wvf in newWvfs:
                    group.add_waveform(wvf)

                result_groups.append(group)

            # ---------------------------------------------------------------------------


        return result_groups


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
            self.card = spcm_hOpen("TCPIP::171.64.57.188::INSTR")  ### remote card mode only for testing purposes
        else:
            self.card = spcm_hOpen(create_string_buffer (b'/dev/spcm0'))
#         if self.card == None:
#             raise LabscriptError("Device is not connected.")

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
            err = spcm_dwSetParam_i32(self.card, SPC_CARDMODE, SPC_REP_STD_MULTI)
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
        results['channel 0']['Power'] = 0
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
        print("Generating buffer")

        # Iterate over the channels which are on. Set channel-specific
        self.num_chs = len(self.channels)
        if (self.num_chs == 3) or (self.num_chs > 4):
            raise LabscriptError("Spectrum card only supports 1, 2, or 4 channels. Please remove a channel or add a dummy channel.")

        channel_enable_word = 0
        for channel in self.channels:

            ## Setting amplitude corresponding to chosen power.
            amplitude = int(np.sqrt(0.1) * 10 ** (float(channel.power) / 20.0) * 1000)
            if amplitude < 80: raise LabscriptError("Power below acceptable range. Min power = -23.9 dBm")
            if amplitude > 2500: raise LabscriptError("Power above acceptable range. Max power = 5.9 dBm.")
            if channel.port == 0:
                channel_enable_word |= CHANNEL0
                err=spcm_dwSetParam_i32(self.card, SPC_AMP0, int32(amplitude))
                if err: raise LabscriptError("Error detected in settings: " + str(err) + "Amplitude: " + str(amplitude))
                err = spcm_dwSetParam_i32(self.card, SPC_ENABLEOUT0, 1)
                if err: raise LabscriptError("Error detected in settings: " + str(err))
            if channel.port == 1:
                channel_enable_word |= CHANNEL1
                err=spcm_dwSetParam_i32(self.card, SPC_AMP1, int32(amplitude))
                if err: raise LabscriptError("Error detected in settings: " + str(err) + "Amplitude: " + str(amplitude))
                err = spcm_dwSetParam_i32(self.card, SPC_ENABLEOUT1, 1)
                if err: raise LabscriptError("Error detected in settings: " + str(err))
            if channel.port == 2:
                channel_enable_word |= CHANNEL2
                err=spcm_dwSetParam_i32(self.card, SPC_AMP2, int32(amplitude))
                if err: raise LabscriptError("Error detected in settings: " + str(err) + "Amplitude: " + str(amplitude))
                err = spcm_dwSetParam_i32(self.card, SPC_ENABLEOUT2, 1)
                if err: raise LabscriptError("Error detected in settings: " + str(err))
            if channel.port == 3:
                channel_enable_word |= CHANNEL3
                err=spcm_dwSetParam_i32(self.card, SPC_AMP3, int32(amplitude))
                if err: raise LabscriptError("Error detected in settings: " + str(err) + "Amplitude: " + str(amplitude))
                err = spcm_dwSetParam_i32(self.card, SPC_ENABLEOUT3, 1)
                if err: raise LabscriptError("Error detected in settings: " + str(err))

#         print(channel_enable_word)
        err = spcm_dwSetParam_i32(self.card, SPC_CHENABLE, int32(channel_enable_word))
        if err: raise LabscriptError("Error detected in settings: " + str(err))

        #### MULTI MODE #### Each segment must be same size.
        if (self.mode == 'multi'):
#             lSetChannels = int32 (0)
#             spcm_dwGetParam_i32 (self.card, SPC_CHCOUNT,     byref (lSetChannels))
#             print('num chs')
#             print(lSetChannels.value)
#             lBytesPerSample = int32 (0)
#             spcm_dwGetParam_i32 (self.card, SPC_MIINST_BYTESPERSAMPLE,  byref (lBytesPerSample))
#             print(lBytesPerSample.value)
#             lFeats = int32 (0)
#             spcm_dwGetParam_i32 (self.card, SPC_PCIFEATURES,  byref (lFeats))
#             print(lFeats.value)

            lStatus = int32 (0)
            spcm_dwGetParam_i32 (self.card, SPC_M2STATUS,  byref (lStatus))
#             print('STATUS------------------------------')
#             print(lStatus.value)


            self.num_groups = len(self.sample_groups)
            max_samples = 0 ### For multimode, we must know the largest size segment, in order to make each segment the same size.
            for group in self.sample_groups:
                num = group.duration * self.clock_freq
                if num > max_samples: max_samples = num
            self.samples = int(max_samples)

            if (self.samples % 32) != 0: raise LabscriptError("Number of samples must be a multiple of 32") # Not *strictly* necessary: see p.105 of Spectrum manual
            size = uint64(self.num_groups * self.num_chs * self.samples * self.bytesPerSample)
            self.buffer = create_string_buffer(size.value)
            print("Filling waveform...")

            np_waveform = np.zeros(self.num_groups * self.num_chs * self.samples, dtype=int16)

            for i,group in enumerate(self.sample_groups):

                for j in range(len(group.segments)):
                    seg = group.segments[j]

                    segSamples = int(seg.duration * self.clock_freq)
                    t = np.linspace(0, seg.duration, segSamples)

                    ramp = np.zeros(segSamples)
                    for pulse in seg.pulses:
                        if pulse.ramp_type == "linear":
                            ramp += pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=seg.duration, f1=pulse.end, method='linear', phi=pulse.phase)
                        elif pulse.ramp_type == "quadratic":
                            ramp += pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=seg.duration, f1=pulse.end, method='quadratic', phi=pulse.phase)
                        else: ## If no allowed ramp is specified, then it is assumed that the frequency remains stationary.
                            ramp += pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=seg.duration, f1=pulse.start, method='linear', phi=pulse.phase)

                    ramp = ramp.astype(int16)

                    groupOffs = int(i * self.samples * self.num_chs)
                    segmOffs = int((seg.time - group.time) * self.clock_freq * self.num_chs)
                    channelOffs = int(seg.port)
                    offset = groupOffs + segmOffs + channelOffs

                    begin = offset
                    end = offset + (len(ramp) * int(self.num_chs))
                    increment = int(self.num_chs)
                    np_waveform[begin:end:increment] = ramp


            memmove(self.buffer, np_waveform.ctypes.data_as(ptr16), size.value)

#             with open('X:\\dataDump.csv', 'w') as f:
#                 for i in range(0, self.num_groups * self.num_chs * self.samples):
#                     print >> f, np_waveform[i]

            print("Waveform filled")


            ## Card settings specific to multimode
#             print(self.samples)
#             print(self.num_groups)
#             print(self.num_chs)
            err=spcm_dwSetParam_i32(self.card, SPC_MEMSIZE, self.samples * self.num_groups)
            if err: raise LabscriptError("Error detected in settings: " + str(err))
            err=spcm_dwSetParam_i32(self.card, SPC_SEGMENTSIZE, self.samples)
            if err: raise LabscriptError("Error detected in settings: " + str(err))
            err=spcm_dwSetParam_i32(self.card, SPC_LOOPS, 1)
            if err: raise LabscriptError("Error detected in settings: " + str(err))

        #### SEQUENCE MODE ####
        elif (self.mode == 'sequence'):
            raise LabscriptError("Mode not currently available.")

        #### SINGLE MODE #### -- useful to loop over one frequency for an extended period of time.
        elif (self.mode == 'single'):
            raise LabscriptError("Mode not currently available.")

#             ch = self.channels[0]
#             if not ch.segments: return False
#             seg = ch.segments[0]
#             self.num_segments = 1
#             self.loops = seg.loops
#             self.samples = int(seg.duration*self.clock_freq)
#             if (self.samples % 32) != 0: raise LabscriptError("Number of samples must be a multiple of 32") # Not *strictly* necessary: see p.105 of Spectrum manual
#             size = uint64(self.samples * self.bytesPerSample)
#             self.buffer = create_string_buffer(size.value)
#             waveform = cast(self.buffer, c_void_p)     # Void pointer so we can do pointer arithmentic below to fill the buffer faster using memmove()
#             print("Filling waveform... single")
#             t = np.linspace(0,seg.duration,self.samples)
#             for pulse in seg.pulses:
#                 if pulse.ramp_type == "linear":
#                     ramp = pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=seg.duration, f1=pulse.end, method='linear', phi=pulse.phase)
#                     ramp = ramp.astype(int16)
#                 elif pulse.ramp_type == "quadratic":
#                     ramp = pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=seg.duration, f1=pulse.end, method='quadratic', phi=pulse.phase)
#                     ramp = ramp.astype(int16)
#                 else:
#                     ramp = pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=seg.duration, f1=pulse.start, method='linear', phi=pulse.phase)
#                     ramp = ramp.astype(int16)

#                 # Copy the data over into the buffer
#                 memmove(waveform.value, ramp.ctypes.get_data(), self.samples*sizeof(int16))

#             print("Waveform filled single")
#             waveform = cast(waveform, ptr16)
            #### RESET AMPLITUDE BASED ON NUMBER OF PULSES ####
            # new_amplitude = amplitude / 3
            # err=spcm_dwSetParam_i32(self.card, SPC_AMP2, int32(new_amplitude))
            # if err: raise LabscriptError("Error detected in settings: " + str(err))

#                 err = spcm_dwSetParam_i32(self.card, SPC_MEMSIZE, self.samples)
#                 if err: raise LabscriptError("Error detected in settings: " + str(err))
#                 err = spcm_dwSetParam_i32(self.card, SPC_LOOPS, uint32(self.loops))
#                 if err:
#                     raise LabscriptError("Error detected in settings: " + str(err))
        return True

    def set_trigger(self):
        if self.mode == 'Off': return

        error = spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER) # | M2CMD_CARD_WAITTRIGGER)
        if error != 0: raise LabscriptError("Error detected during data transfer to card. Error: " + str(error))
        # dwError = spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_WAITREADY)

    def transfer_buffer(self):
        print("Transferring data to card...")
        if self.mode == 'Off': return
        dw = spcm_dwDefTransfer_i64 (self.card, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32 (0), self.buffer, uint64 (0), uint64(self.num_groups * self.num_chs * self.samples * self.bytesPerSample))
        dwError = spcm_dwSetParam_i32 (self.card, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
        if((dw + dwError) != 0): raise LabscriptError("Error detected during data transfer to card. Error: " + str(dw) + " " + str(dwError))
        self.buffer = None     # Set to None so the garbage collector can get rid of the buffer when we're done
        gc.collect()           # Tell the garbage collector to throw away the buffer data (we get a memory leak if we don't explicitly do this)
        print("Transfer complete")

    def transition_to_buffered(self,device_name,h5file,initial_values,fresh):
        # return self.final_values
        self.waveform_groups = []
        with h5py.File(h5file) as file:
            device = file['/devices/' +device_name]

            settings = device['device_settings']
            self.mode = settings['mode']
            self.clock_freq = MEGA(int(settings['clock_freq']))

            ch_settings = device['channel_settings'][:]
            self.channels = []
            for i, channel in enumerate(ch_settings):
                self.channels.append(channel_settings(channel['name'], channel['power'], channel['port']))

            groups_folder = device['waveform_groups']

            for groupname in groups_folder.iterkeys():
                g = groups_folder[groupname]
                gsettings = g['group_settings']
                time = gsettings['time']
                duration = gsettings['duration']
                loops = gsettings['loops']
                waveforms = []

                for wavename in g.iterkeys():
                    if wavename != 'group_settings':
                        wvf = waveform(0, 0, 0)
                        s = g[wavename]
                        for p in s.iterkeys():
                            if p == 'waveform_settings':
                                wvf.time = s['waveform_settings']['time'][0]
                                wvf.duration = s['waveform_settings']['duration'][0]
                                wvf.loops = s['waveform_settings']['loops'][0]
                                wvf.port = s['waveform_settings']['port'][0]
                                wvf.sample_start = s['waveform_settings']['sample_start'][0]
                                wvf.sample_end = s['waveform_settings']['sample_end'][0]
                            if p == 'pulse_data':
                                dset = s['pulse_data']
                                ###### Figure out how to go through every line of the profile table.
                                for i in range(dset.shape[0]):
                                    start_freq = dset['start_freq'][i]
                                    end_freq = dset['end_freq'][i]
                                    phase = dset['phase'][i]
                                    amplitude = dset['amp'][i]
                                    ramp_type = dset['ramp_type'][i]
                                    wvf.add_pulse(start_freq, end_freq, phase, amplitude,ramp_type)
                        waveforms.append(wvf)

                self.waveform_groups.append(waveform_group(time,duration,waveforms,loops))

            if len(self.waveform_groups) == 0:
                raise LabscriptError("Did not find any sample groups. Either something is wrong, or you haven't instructed the Spectrum card to do anything.")


            # Sort groups in time order (just in case)
            self.waveform_groups = sorted(self.waveform_groups, key=lambda k: k.time)

            # Plot graphic tool
            fig, ax = plt.subplots(1)
            self.make_segment_rects(self.waveform_groups, ax)
            plt.show()

        buf_gen = self.generate_buffer()
        if buf_gen is True:
            ### Card Settings ###
            self.card_settings()

            ### Send Buffer and Set Trigger ###
            self.transfer_buffer()
            self.set_trigger()

        return self.final_values ### NOT CURRENTLY USED FOR ANY INFO TO SEND TO GUI. NECESSARY TO WORK WITH BLACS.


    def make_segment_rects(self, sample_groups, ax):

        # Create list for all the error patches
        segm_rects = []
        group_rects = []

        last_t = 0

        time_conv = 64e-9     # Convert from sample chunks to seconds

        for i,group in enumerate(sample_groups):

            for k in range(group.loops):
                group_rect = Rectangle(((group.time + k * group.duration)*time_conv, 0), group.duration*time_conv, 4)
                group_rects.append(group_rect)

                if k == 0:
                    text = 'group ' + str(i)
                    if group.loops > 1:
                        text += (' (x' + str(group.loops) + ')')
                    ax.text((group.time + (group.loops * group.duration * 0.5))*time_conv,-0.2,text,ha='center')
                    ax.axvline((group.time + k * group.duration)*time_conv,color='k')

                if group.time + (k+1) * group.duration > last_t:
                    last_t = group.time + (k+1) * group.duration

                for j,waveform in enumerate(group.waveforms):
                    for m in range(waveform.loops):
                        segm_rect = Rectangle(((waveform.time + k * group.duration + m * waveform.duration)*time_conv, waveform.port), waveform.duration*time_conv, 1)
                        segm_rects.append(segm_rect)

                        if k == 0 and m == 0:
                            s_text = 's' + str(j)
                            if waveform.loops > 1:
                                s_text += (' (x' + str(waveform.loops) + ')')
                            ax.text((waveform.time + (0.5 * waveform.loops * waveform.duration))*time_conv,waveform.port+0.5,s_text,ha='center')

        # Create patch collection with specified colour/alpha
        group_pc = PatchCollection(group_rects, facecolor='lightgray', alpha=1, edgecolor='gray')
        segm_pc = PatchCollection(segm_rects, facecolor='r', alpha=0.5, edgecolor='r')

        # Add collection to axes
        ax.add_collection(group_pc)
        ax.add_collection(segm_pc)

        ax.set_xlim(0,last_t * time_conv * 1.02)
        ax.set_ylim(-0.5,4.2)

        ax.set_yticks([0.5,1.5,2.5,3.5],minor=True)
        ax.set_yticklabels([0,1,2,3],minor=True)
        ax.set_yticklabels(['','','','','','','','','',''],minor=False)
        ax.tick_params(axis='y', which='minor', length = 0)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel')


    ### Other Functions, manual mode is not utilized for the spectrum instrumentation card. ###
    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)
    def abort_buffered(self):
        return self.transition_to_manual(True)
    def transition_to_manual(self,abort = False):
        if abort:
            self.shutdown()
            self.init()
#         else: spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_STOP)
        return True
    def shutdown(self):
        spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_STOP)
        spcm_vClose(self.card)
