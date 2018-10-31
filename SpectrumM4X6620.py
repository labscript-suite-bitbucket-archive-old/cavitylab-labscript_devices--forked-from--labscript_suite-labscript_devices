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
import binascii

##### All the data for the sequence is wrapped up in these class structures. #####
class pulse():
    def __init__(self,start_freq,end_freq,ramp_time,phase,amplitude,ramp_type):
        self.start = start_freq
        self.end = end_freq
        self.ramp_time = ramp_time  # In seconds
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
            self.pulses.append(pulse(p.start,p.end,p.ramp_time,p.phase,p.amp,p.ramp_type))

        self.sample_start = 0
        self.sample_end = duration

    def add_pulse(self,start_freq,end_freq,ramp_time,phase,amplitude,ramp_type):
        self.pulses.append(pulse(start_freq,end_freq,ramp_time,phase,amplitude,ramp_type))

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

class sequence_instr():
    def __init__(self,step,next_step,segment,loops):
        self.step = step
        self.segment = segment
        self.loops = loops
        self.next_step = next_step


# Convert from time in seconds to time in sample chunks (1 chunk = 32 samples)
# clock_freq in Hz
def time_s_to_c(t, clock_freq):
    return int(math.floor(float(t * clock_freq) / 32.0))

# Convert from time in sample chunks to time in seconds (1 chunk = 32 samples)
# clock_freq in Hz
def time_c_to_s(t, clock_freq):
    return float(t * 32.0) / float(clock_freq)


def draw_waveform_groups(waveform_groups, clock_freq, ax):

    segm_rects = []
    group_rects = []

    last_t = 0

    for i,group in enumerate(waveform_groups):

        for k in range(group.loops):
            group_rect = Rectangle((time_c_to_s(group.time + k * group.duration, clock_freq), 0), time_c_to_s(group.duration, clock_freq), 4)
            group_rects.append(group_rect)

            if k == 0:
                text = 'group ' + str(i)
                if group.loops > 1:
                    text += (' (x' + str(group.loops) + ')')
                ax.text(time_c_to_s(group.time + (group.loops * group.duration * 0.5), clock_freq),-0.2,text,ha='center')
                ax.axvline(time_c_to_s(group.time + k * group.duration, clock_freq),color='k')

            if group.time + (k+1) * group.duration > last_t:
                last_t = group.time + (k+1) * group.duration

            for j,waveform in enumerate(group.waveforms):
                for m in range(waveform.loops):
                    segm_rect = Rectangle((time_c_to_s(waveform.time + k * group.duration + m * waveform.duration, clock_freq), waveform.port), time_c_to_s(waveform.duration, clock_freq), 1)
                    segm_rects.append(segm_rect)

                    if k == 0 and m == 0:
                        s_text = 's' + str(j)
                        if waveform.loops > 1:
                            s_text += (' (x' + str(waveform.loops) + ')')
                        ax.text(time_c_to_s(waveform.time + (0.5 * waveform.loops * waveform.duration), clock_freq),waveform.port+0.5,s_text,ha='center')

    # Create patch collection with specified colour/alpha
    group_pc = PatchCollection(group_rects, facecolor='lightgray', alpha=1, edgecolor='gray')
    segm_pc = PatchCollection(segm_rects, facecolor='r', alpha=0.5, edgecolor='r')

    # Add collection to axes
    ax.add_collection(group_pc)
    ax.add_collection(segm_pc)

    ax.set_xlim(0, time_c_to_s(last_t * 1.02, clock_freq))
    ax.set_ylim(-0.5,4.2)

    ax.set_yticks([0.5,1.5,2.5,3.5],minor=True)
    ax.set_yticklabels([0,1,2,3],minor=True)
    ax.set_yticklabels(['','','','','','','','','',''],minor=False)
    ax.tick_params(axis='y', which='minor', length = 0)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    # ax.title('Waveform Groups')

    return


def draw_sequence_plot(waveform_groups, sequence_instrs, clock_freq, ax):

    segm_rects = []
    group_rects = []
    dummy_group_rects = []

    last_t = 0

    seq_index = 0
    cur_time = 0
    while True:
        seq_instr = sequence_instrs[seq_index]
        group = waveform_groups[seq_instr.segment]

        if group.waveforms == 'dummy' and seq_instr.segment == 0:
            group_rect = Rectangle((time_c_to_s(cur_time, clock_freq), 0), time_c_to_s(group.duration * seq_instr.loops, clock_freq), 4)
            dummy_group_rects.append(group_rect)

            text = 'group ' + str(seq_instr.segment)
            if seq_instr.loops > 1:
                text += (' (x' + str(seq_instr.loops) + ')')
            ax.text(time_c_to_s(cur_time + (seq_instr.loops * group.duration * 0.5), clock_freq),-0.2,text,ha='center')
            ax.axvline(time_c_to_s(cur_time, clock_freq),color='k')

            cur_time += group.duration * seq_instr.loops

        else:
            for k in range(seq_instr.loops):
                group_rect = Rectangle((time_c_to_s(cur_time, clock_freq), 0), time_c_to_s(group.duration, clock_freq), 4)
                group_rects.append(group_rect)

                if k == 0:
                    text = 'group ' + str(seq_instr.segment)
                    if seq_instr.loops > 1:
                        text += (' (x' + str(seq_instr.loops) + ')')
                    ax.text(time_c_to_s(cur_time + (seq_instr.loops * group.duration * 0.5), clock_freq),-0.2,text,ha='center')
                    ax.axvline(time_c_to_s(cur_time, clock_freq),color='k')

                if cur_time + group.duration > last_t:
                    last_t = cur_time + group.duration

                if group.waveforms != 'dummy':
                    for j,waveform in enumerate(group.waveforms):
                        for m in range(waveform.loops):
                            segm_rect = Rectangle((time_c_to_s(cur_time + waveform.time - group.time + m * waveform.duration, clock_freq), waveform.port), time_c_to_s(waveform.duration, clock_freq), 1)
                            segm_rects.append(segm_rect)

                            if k == 0 and m == 0:
                                s_text = 's' + str(j)
                                if waveform.loops > 1:
                                    s_text += (' (x' + str(waveform.loops) + ')')
                                ax.text(time_c_to_s(waveform.time + (0.5 * waveform.loops * waveform.duration), clock_freq),waveform.port+0.5,s_text,ha='center')

            cur_time += group.duration

        seq_index = seq_instr.next_step
        if seq_index == 0:
            break

    print('Drawing seq plot')

    # Create patch collection with specified colour/alpha
    group_pc = PatchCollection(group_rects, facecolor='lightgray', alpha=1, edgecolor='gray')
    dummy_group_pc = PatchCollection(dummy_group_rects, facecolor='gray', alpha=1, edgecolor='gray')
    segm_pc = PatchCollection(segm_rects, facecolor='r', alpha=0.5, edgecolor='r')

    # Add collection to axes
    ax.add_collection(group_pc)
    ax.add_collection(dummy_group_pc)
    ax.add_collection(segm_pc)

    ax.set_xlim(0, time_c_to_s(last_t * 1.02, clock_freq))
    ax.set_ylim(-0.5,4.2)

    ax.set_yticks([0.5,1.5,2.5,3.5],minor=True)
    ax.set_yticklabels([0,1,2,3],minor=True)
    ax.set_yticklabels(['','','','','','','','','',''],minor=False)
    ax.tick_params(axis='y', which='minor', length = 0)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    # ax.title('Sequence')

    return



@labscript_device
class SpectrumM4X6620(IntermediateDevice):

    def __init__(self,name,parent_device,trigger,triggerDur=100e-6):
        IntermediateDevice.__init__(self,name,parent_device)
        self.BLACS_connection = 5
        self.set_mode('Off') ## Initialize data structure

        self.triggerDur = triggerDur

        self.raw_waveforms = []

        if trigger:
            if 'device' in trigger and 'connection' in trigger:
                self.triggerDO = DigitalOut(self.name+'_Trigger', trigger['device'], trigger['connection'])
            else:
                raise LabscriptError('You must specify the "device" and "connection" for the trigger input to the SpectrumM4X6620')
        else:
            raise LabscriptError('No trigger specified for device ' + self.name)

    ## Sets up the channel_data structure that will be filled with the following function calls (single_freq,comb,sweep...).
    def set_mode(self, mode_name, channels=[], clock_freq=500, use_ext_clock=True, ext_clock_freq=10):
        self.use_ext_clock = use_ext_clock
        self.ext_clock_freq = MEGA(ext_clock_freq)

        if (len(channels) == 3 or len(channels) > 4):
            raise LabscriptError('SpectrumM4X6620 only supports 1, 2, or 4 channels. Please remove a channel or add a dummy channel')

        channel_objects = []
        for i,channel in enumerate(channels):
            if not (channel['name'] == '' or channel['name'] == None):
                channel_objects.append(channel_settings(channel['name'],channel['power'],i))

        self.sample_data = sample_data(channels=channel_objects,mode=mode_name,clock_freq=MEGA(clock_freq))

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

        if t < 0:
            raise LabscriptError('Time t cannot be negative')
        if duration <= 0:
            raise LabscriptError('Duration must be positive')
        if duration > 100e-3:
            sys.stderr.write('Warning: duration of waveform is very long. You may run out of memory on the experiment control computer or on the Spectrum card.\n')
        if duration > 3.3:
            raise LabscriptError('Waveform duration exceeds card memory (3.3s)')

        # Convert from time in seconds to time in sample chunks (1 sample chunk = 32 samples)
        t_c = time_s_to_c(t, self.sample_data.clock_freq)
        duration_c = time_s_to_c(duration, self.sample_data.clock_freq)

        if loops > 2**32 - 1:
            raise LabscriptError('Too many loops requested. Number of loops must be less than 2^32')
        if duration_c < 12:
            raise LabscriptError('Duration of segment is too short. Segment duration must be at least 12 sample chunks ( = 768ns)')

        wvf = waveform(t_c,duration_c,ch,loops,is_periodic=(loops > 1))

        for i in range(len(start_freqs)):
            if (amplitudes[i] < 0) or (amplitudes[i] > 1):
                raise LabscriptError("Amplitude[" + str(i) + "] = " + str(amplitudes[i]) + " is outside the allowed range [0,1]")

            wvf.add_pulse(start_freqs[i],end_freqs[i],duration,phases[i],amplitudes[i],ramp_type)
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

        device = hdf5_file.create_group('/devices/' + self.name)

        # Store device settings
        settings_dtypes = np.dtype([('mode', 'S10'),('clock_freq',np.float),('use_ext_clock',np.int),('ext_clock_freq',np.float)])
        settings_table = np.array((0,0,0,0),dtype=settings_dtypes)
        settings_table['mode'] = self.sample_data.mode
        settings_table['clock_freq'] = self.sample_data.clock_freq
        settings_table['use_ext_clock'] = self.use_ext_clock
        settings_table['ext_clock_freq'] = self.ext_clock_freq
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

            if group.duration == 0:
                raise LabscriptError('Something went wrong in preparing waveform data. Group duration is 0')

            # Store waveforms
            for wvf in group.waveforms:
                name = "Waveform: ch = " + str(wvf.port) + ", t = " + str(wvf.time) + ", dur = " + str(wvf.duration)
                if name in group_folder:   ## If waveform already exists, add to already created group
                    grp = group_folder[name]
                else:
                    grp = group_folder.create_group(name)
                    profile_dtypes = [('time', int), ('duration', int), ('loops', int), ('port', int), ('sample_start', int), ('sample_end', int)]
                    profile_table = np.zeros(1, dtype=profile_dtypes)
                    profile_table['time'] = wvf.time
                    profile_table['duration'] = wvf.duration
                    profile_table['loops'] = wvf.loops
                    profile_table['port'] = wvf.port
                    profile_table['sample_start'] = wvf.sample_start
                    profile_table['sample_end'] = wvf.sample_end
                    grp.create_dataset('waveform_settings', data=profile_table)

                if wvf.duration == 0:
                    raise LabscriptError('Something went wrong in preparing waveform data. Waveform duration is 0')

                # Store pulses
                profile_dtypes = [('start_freq', np.float),
                                  ('end_freq', np.float),
                                  ('ramp_time', np.float),
                                  ('phase', np.float),
                                  ('amp', np.float),
                                  ('ramp_type',"S10")]
                profile_table = np.zeros(len(wvf.pulses), dtype=profile_dtypes)


                if len(wvf.pulses) == 0:
                    raise LabscriptError('Something went wrong in generating Spectrum card data: waveform does not have any pulse data')

                for j,pulse in enumerate(wvf.pulses):
                    profile_table['start_freq'][j] = pulse.start
                    profile_table['end_freq'][j] = pulse.end
                    profile_table['ramp_time'][j] = pulse.ramp_time
                    profile_table['phase'][j] = pulse.phase
                    profile_table['amp'][j] = pulse.amp
                    profile_table['ramp_type'][j] = pulse.ramp_type

                if 'pulse_data' in grp: ### If waveform already has associated data, add to the existing dataset.
                    d = grp['pulse_data']
                    d.resize((d.shape[0]+profile_table.shape[0]), axis=0)
                    d[-profile_table.shape[0]:] = profile_table
                else:
                    grp.create_dataset('pulse_data', maxshape=(1000,),
                                       data=profile_table, dtype = profile_dtypes, chunks = True)


    def stop(self):
        self.check_channel_collisions(self.raw_waveforms)

        if self.sample_data.mode == 'multi':
            self.sample_data.waveform_groups = self.make_waveform_groups(self.raw_waveforms)

            # Sort groups in time order (just in case)
            self.sample_data.waveform_groups = sorted(self.sample_data.waveform_groups, key=lambda k: k.time)

            max_dur = 0
            for group in self.sample_data.waveform_groups:
                if group.duration > max_dur:
                    max_dur = group.duration

            # Fire the trigger at the appropriate times, and check that the timing works out
            t_prev = float('-inf')
            for group in self.sample_data.waveform_groups:
                self.triggerDO.go_high(time_c_to_s(group.time, self.sample_data.clock_freq))
                self.triggerDO.go_low(time_c_to_s(group.time, self.sample_data.clock_freq)+self.triggerDur)

                if self.triggerDur >= time_c_to_s(group.duration, self.sample_data.clock_freq):
                    raise LabscriptError("Trigger duration too long (i.e. longer than the sample group duration; might not be a problem depending on how the card interprets trigger edges)")

                if group.time - t_prev < max_dur:
                    raise LabscriptError("Maximum group length is larger than the separation between groups! This is an edge case that Greg hoped wouldn't crop up but apparently it has (see Cavity Lab wiki page on 9/25/2018).")

                t_prev = group.time

        elif self.sample_data.mode == 'sequence':
            periodicWvfs = filter(lambda k: k.is_periodic == True, self.raw_waveforms)
            nonPeriodicWvfs = filter(lambda k: k.is_periodic == False, self.raw_waveforms)

            nonPeriodicWvfGroups = self.make_waveform_groups(nonPeriodicWvfs)

            self.sample_data.waveform_groups = self.combine_periodic_nonperiodic_groups(periodicWvfs, nonPeriodicWvfGroups)

            # Sort groups in time order (just in case)
            self.sample_data.waveform_groups = sorted(self.sample_data.waveform_groups, key=lambda k: k.time)

            # Initial trigger
            self.triggerDO.go_high(0)
            self.triggerDO.go_low(self.triggerDur)

        else:
            raise LabscriptError('Unrecognized mode.')

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

            t_n = int(t0 + math.ceil(float(t_start_p - t0) / float(dur)) * dur)     # Start of the next loop immediately following t_start_p
            n_full_loops = int(math.floor(float(t_end_p - t_n) / float(dur)))

            if t_n > t_end_p: # Partial loop is completely inside the desired region
                # Partial loop
                waveform_partial = waveform(t_start_p,t_end_p-t_start_p,wvf.port,loops=1,is_periodic=False,pulses=wvf.pulses)
                waveform_partial.sample_start = t_start_p - (t_n - dur)
                waveform_partial.sample_end = waveform_partial.sample_start + t_end_p - t_start_p
                result_waveforms.append(waveform_partial)

            else:
                # Full middle loops
                if n_full_loops > 0:
                    waveform_full_loops = waveform(t_n,dur,wvf.port,loops=n_full_loops,is_periodic=True,pulses=wvf.pulses)
                    result_waveforms.append(waveform_full_loops)

                # Partial start loop
                if t_start_p != t_n:
                    dt_start = t_start_p - (t_n - dur)
                    waveform_partial_start = waveform(t_start_p,dur-dt_start,wvf.port,loops=1,is_periodic=False,pulses=wvf.pulses)
                    waveform_partial_start.sample_start = dt_start
                    waveform_partial_start.sample_end = dur
                    result_waveforms.append(waveform_partial_start)

                # Partial end loop
                if t_end_p != (t_n + n_full_loops * dur):
                    dt_end = t_end_p - (t_n + n_full_loops * dur)
                    waveform_partial_end = waveform(t_n+(n_full_loops * dur),dt_end,wvf.port,loops=1,is_periodic=False,pulses=wvf.pulses)
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
        self.remote = True
        if self.remote:
            self.card = spcm_hOpen("TCPIP::171.64.58.213::INSTR")  ### remote card mode only for testing purposes
        else:
            self.card = spcm_hOpen(create_string_buffer (b'/dev/spcm0'))
        if self.card == None:
            raise LabscriptError("Device is not connected.")

        self.samplesPerChunk = 32
        self.bytesPerSample = 2


    def card_settings(self):
#         err_text = c_char_p('')
#         spcm_dwGetErrorInfo_i32(self.card, 0, 0, err_text)
#         print(err_text)

        ## General settings -- mode specific settings are defined in transition_to_buffered
        if self.use_ext_clock == True:
            err=spcm_dwSetParam_i32(self.card, SPC_CLOCKMODE, SPC_CM_EXTREFCLOCK)  # clock mode external PLL
            if err: raise LabscriptError("Error detected in settings: " + str(err))
            err=spcm_dwSetParam_i32(self.card, SPC_REFERENCECLOCK, self.ext_clock_freq)
            if err: raise LabscriptError("Error detected in settings: " + str(err))
        else:
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
            err = spcm_dwSetParam_i32(self.card, SPC_CARDMODE, SPC_REP_STD_SEQUENCE)
            if err: raise LabscriptError("Error detected in settings: " + str(err))
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

            self.num_groups = len(self.waveform_groups)
            max_samples = 0 ### For multimode, we must know the largest size segment, in order to make each segment the same size.
            for group in self.waveform_groups:
                num = group.duration * self.samplesPerChunk
                if num > max_samples: max_samples = num
            self.samples = int(max_samples)

            if (self.samples % 32) != 0:
                # raise LabscriptError("Number of samples must be a multiple of 32") # Not *strictly* necessary: see p.105 of Spectrum manual
                print('Warning: number of samples must be a multiple of 32. Rounding up number of samples...')
                self.samples += 32 - (self.samples % 32)

            size = uint64(self.num_groups * self.num_chs * self.samples * self.bytesPerSample)
            self.buffer = create_string_buffer(size.value)
            print("Filling waveform...")

            np_waveform = np.zeros(self.num_groups * self.num_chs * self.samples, dtype=int16)

            for i,group in enumerate(self.waveform_groups):

                for j,wvf in enumerate(group.waveforms):

                    wvfSamples = int(wvf.duration * self.samplesPerChunk)
                    t = np.linspace(0, time_c_to_s(wvf.duration, self.clock_freq), wvfSamples)

                    ramp = np.zeros(wvfSamples)
                    for pulse in wvf.pulses:
                        if pulse.ramp_type == "linear":
                            ramp += pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=pulse.ramp_time, f1=pulse.end, method='linear', phi=pulse.phase)
                        elif pulse.ramp_type == "quadratic":
                            ramp += pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=pulse.ramp_time, f1=pulse.end, method='quadratic', phi=pulse.phase)
                        else: ## If no allowed ramp is specified, then it is assumed that the frequency remains stationary.
                            ramp += pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=pulse.ramp_time, f1=pulse.start, method='linear', phi=pulse.phase)

                    ramp = ramp.astype(int16)

                    groupOffs = int(i * self.samples * self.num_chs)
                    wvfmOffs = int((wvf.time - group.time) * self.samplesPerChunk * self.num_chs)
                    channelOffs = int(wvf.port)
                    offset = groupOffs + wvfmOffs + channelOffs

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

            # Sort groups in time order (just in case)
            self.waveform_groups = sorted(self.waveform_groups, key=lambda k: k.time)

            # Add dummy sequences between waveform groups
            dummy_groups = []
            self.sequence_instrs = []

            # Add main dummy loop segment to beginning of stack
            # If dummy segments are too short, we can't do enough loops to last for many seconds of
            # downtime (we are limited to at most 2^20 loops). So we must loop over longer dummy segments
            dummy_loop_dur = 1024     # length in segment chunks (1 chunk = 32 segments)
            dummy_groups.append(waveform_group(float('-inf'),dummy_loop_dur,'dummy'))


            # Add leading dummy groups and generate sequence instructions
            cur_step = 0
            cur_segm = 1     # segm 0 is the dummy loop
            for i,group in enumerate(self.waveform_groups):

                # t0 = end of previous group, t1 = start of this group
                if i == 0:
                    t0 = 0
                else:
                    prev_group = self.waveform_groups[i-1]
                    t0 = prev_group.time + prev_group.loops * prev_group.duration

                t1 = group.time

                # Play leading dummy segment
                dummy_dur = t1 - t0
                if dummy_dur != 0:
                    if dummy_dur > 12:     # If dummy duration is longer than the minimum segment size
                        n_loops = int(math.floor(float(dummy_dur) / float(dummy_loop_dur)))
                        leftover = dummy_dur - dummy_loop_dur * n_loops

                        if n_loops > 0:
                            # Send card to segment 0
                            self.sequence_instrs.append(sequence_instr(cur_step,cur_step+1,0,n_loops))
                            cur_step+=1

                        if leftover > 0:
                            # Send card to the 'leftover' dummy segment
                            self.sequence_instrs.append(sequence_instr(cur_step,cur_step+1,cur_segm,1))
                            cur_step+=1
                            cur_segm+=1
                            dummy_groups.append(waveform_group(t0 + dummy_loop_dur * n_loops,leftover,'dummy'))

                    else:                 # Extend group backward so that it starts at t0
                        group.time = t0

                # Play group segment
                self.sequence_instrs.append(sequence_instr(cur_step,cur_step+1,cur_segm,group.loops))
                cur_step+=1
                cur_segm+=1

            # Loop the sequence back to the zeroth step
            self.sequence_instrs[len(self.sequence_instrs)-1].next_step = 0

            self.waveform_groups.extend(dummy_groups)

            # Sort groups in time order
            self.waveform_groups = sorted(self.waveform_groups, key=lambda k: k.time)

            # Get rid of the '-inf's that we used earlier
            # And check for zero-duration groups
            for group in self.waveform_groups:
                group.time = max(group.time,0)

                if group.duration == 0:
                    raise LabscriptError('Something went wrong in preparing waveform data. Group duration is 0')


            # for group in self.waveform_groups:
            #     print('--------------')
            #     print(time_c_to_s(group.time, self.clock_freq))
            #     print(group.waveforms)
            #     print(time_c_to_s(group.time + group.duration * group.loops, self.clock_freq))
            #
            # for instr in self.sequence_instrs:
            #     print('**************')
            #     print(instr.step)
            #     print(instr.next_step)
            #     print(instr.segment)
            #     print(instr.loops)
            #     if instr.segment == 0:
            #         print(time_c_to_s(instr.loops * dummy_loop_dur,self.clock_freq))


            samples_per_chunk = self.samplesPerChunk
            bytes_per_sample = self.bytesPerSample

            # Split memory into segments
            num_segments = len(self.waveform_groups) + 1
            num_segments = int(2**math.ceil(math.log(num_segments,2)))
            spcm_dwSetParam_i32(self.card, SPC_SEQMODE_MAXSEGMENTS, int32(num_segments))
            spcm_dwSetParam_i32(self.card, SPC_SEQMODE_STARTSTEP, 0)

            # Write segments
            for i,group in enumerate(self.waveform_groups):

                buffer_size = int(self.num_chs * group.duration * samples_per_chunk * bytes_per_sample)
                pBuffer = create_string_buffer(buffer_size)

                np_waveform = np.zeros(self.num_chs * group.duration * samples_per_chunk, dtype=int16)

                # Fill buffer
                if group.waveforms != 'dummy':

                    for wvf in group.waveforms:
                        dur = wvf.sample_end - wvf.sample_start

                        pulse_data = np.zeros(dur * samples_per_chunk)

                        t = np.linspace(time_c_to_s(wvf.sample_start, self.clock_freq), time_c_to_s(wvf.sample_end, self.clock_freq), dur * samples_per_chunk)

                        for pulse in wvf.pulses:
                            if pulse.ramp_type == "linear":
                                pulse_data += pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=pulse.ramp_time, f1=pulse.end, method='linear', phi=pulse.phase)
                            elif pulse.ramp_type == "quadratic":
                                pulse_data += pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=pulse.ramp_time, f1=pulse.end, method='quadratic', phi=pulse.phase)
                            else: ## If no allowed ramp is specified, then it is assumed that the frequency remains stationary.
                                pulse_data += pulse.amp * (2**15-1) * chirp(t, f0=pulse.start, t1=pulse.ramp_time, f1=pulse.start, method='linear', phi=pulse.phase)

                        pulse_data = pulse_data.astype(int16)

                        begin = (wvf.time - group.time) * samples_per_chunk * int(self.num_chs) + wvf.port
                        end = begin + (len(pulse_data) * int(self.num_chs))
                        increment = int(self.num_chs)
                        np_waveform[begin:end:increment] = pulse_data

                memmove(pBuffer, np_waveform.ctypes.data_as(ptr16), buffer_size)

                # Write buffer
                err = spcm_dwSetParam_i32(self.card, SPC_SEQMODE_WRITESEGMENT, int32(i))
                if err: raise LabscriptError("Error detected in settings: " + str(err))
                #print('Write segment = '+str(i)+' '+str(type(i)))
                err = spcm_dwSetParam_i32(self.card, SPC_SEQMODE_SEGMENTSIZE, int32(group.duration * samples_per_chunk))
                if err: raise LabscriptError("Error detected in settings: " + str(err))
                #print('Segment size = '+str(group.duration * samples_per_chunk)+' '+str(type(group.duration * samples_per_chunk)))

                dw = spcm_dwDefTransfer_i64(self.card, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32 (0), pBuffer, uint64 (0), uint64 (self.num_chs * group.duration * samples_per_chunk * bytes_per_sample))
                #print('Def Transf = '+str(uint64 (group.duration * samples_per_chunk * bytes_per_sample))+' '+str(type(uint64 (group.duration * samples_per_chunk * bytes_per_sample))))
                err = spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
                if err: raise LabscriptError("Error detected in settings: " + str(err))


            # Write sequence instructions
            for i,instr in enumerate(self.sequence_instrs):
                if i+1 == len(self.sequence_instrs):
                    lCond = SPCSEQ_END
                else:
                    lCond = SPCSEQ_ENDLOOPALWAYS

                lVal = uint64((lCond << 32) | (int(instr.loops) << 32) | (int(instr.next_step) << 16) | (int(instr.segment)))
                err = spcm_dwSetParam_i64(self.card, SPC_SEQMODE_STEPMEM0 + int(instr.step), lVal)
                #print('Seq step = '+str(instr.step)+' '+str(type(instr.step)))
                #print('Seq val = '+str(binascii.hexlify(lVal))+' '+str(type(lVal)))
                if err: raise LabscriptError("Error detected in settings: " + str(err))


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
        if error != 0: raise LabscriptError("Error detected during card start. Error: " + str(error))

        print('Card started')
        # dwError = spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_WAITREADY)

    def transfer_buffer(self):
#         print("Transferring data to card...")
#         if self.mode == 'Off': return
#         dw = spcm_dwDefTransfer_i64 (self.card, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32 (0), self.buffer, uint64 (0), uint64(self.num_groups * self.num_chs * self.samples * self.bytesPerSample))
#         dwError = spcm_dwSetParam_i32 (self.card, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
#         if((dw + dwError) != 0): raise LabscriptError("Error detected during data transfer to card. Error: " + str(dw) + " " + str(dwError))
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
            self.clock_freq = int(settings['clock_freq'])
            self.use_ext_clock = bool(settings['use_ext_clock'])
            self.ext_clock_freq = int(settings['ext_clock_freq'])

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
                                    ramp_time = dset['ramp_time'][i]
                                    phase = dset['phase'][i]
                                    amplitude = dset['amp'][i]
                                    ramp_type = dset['ramp_type'][i]
                                    wvf.add_pulse(start_freq, end_freq, ramp_time, phase, amplitude,ramp_type)
                        waveforms.append(wvf)

                self.waveform_groups.append(waveform_group(time,duration,waveforms,loops))

            if len(self.waveform_groups) == 0:
                raise LabscriptError("Did not find any sample groups. Either something is wrong, or you haven't instructed the Spectrum card to do anything.")


            # Sort groups in time order (just in case)
            self.waveform_groups = sorted(self.waveform_groups, key=lambda k: k.time)

            # Plot graphic tool
            fig, ax = plt.subplots(1)
            draw_waveform_groups(self.waveform_groups, self.clock_freq, ax)
            plt.show()

        ### Card Settings ###
        self.card_settings()

        buf_gen = self.generate_buffer()
        if buf_gen is True:
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
#         else: spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_STOP)
        return True
    def shutdown(self):
        spcm_dwSetParam_i32(self.card, SPC_M2CMD, M2CMD_CARD_STOP)
        spcm_vClose(self.card)
