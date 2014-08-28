import sys
import scipy.io.wavfile
import scipy.signal
import numpy
import numpy.fft
import matplotlib.pyplot as plt
import collections

################################################################################

def plot_dft_samples(samples, sample_rate, title=""):
    samples_dft_mag = 20*numpy.log10(numpy.abs(numpy.fft.fft(samples)))
    samples_freqs = numpy.fft.fftfreq(len(samples_dft_mag), d=1.0/sample_rate)
    plt.plot(samples_freqs, samples_dft_mag)
    plt.xlim([-9000, 9000])
    plt.ylabel('Amplitude (Log)')
    plt.xlabel('Frequency (Hz)')
    plt.title(title)
    plt.show()

def plot_filter(b, a, sample_rate, freqs=None, title=""):
    if freqs is None:
        w, h = scipy.signal.freqz(b, a)
    else:
        w, h = scipy.signal.freqz(b, a, (2*numpy.pi*numpy.array(freqs))/sample_rate)
    plt.plot((sample_rate * w)/(2*numpy.pi), 20*numpy.log10(abs(h)))
    plt.ylabel('Amplitude (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.title(title)
    plt.show()

################################################################################

SAMPLE_RATE = 0.0

################################################################################

def stream_wave_file(path, start=None, stop=None):
    global SAMPLE_RATE

    (SAMPLE_RATE, samples) = scipy.io.wavfile.read(path, mmap=True)

    if start is not None and stop is None:
        samples = samples[start*SAMPLE_RATE:]
    elif start is not None and stop is not None:
        samples = samples[start*SAMPLE_RATE:stop*SAMPLE_RATE]

    return samples

def stream_bandpass_filter_iir(stream, fLow, fHigh, margin=10.0):
    b,a = scipy.signal.butter(3, [(2*fLow)/(SAMPLE_RATE), (2*fHigh)/(SAMPLE_RATE)], btype='bandpass')
    #plot_filter(b, a, SAMPLE_RATE, range(1000))
    return scipy.signal.lfilter(b, a, list(stream))

def stream_rectify(stream):
    for sample in stream:
        yield abs(sample)

def stream_lowpass_filter_iir(stream, fC):
    b, a = scipy.signal.butter(4, (2*fC)/SAMPLE_RATE)
    #plot_filter(b, a, SAMPLE_RATE, range(1000))
    return scipy.signal.lfilter(b, a, list(stream))

def stream_find_threshold(stream):
    samples = list(stream)

    values, positions = numpy.histogram(samples, bins=100)
    sorted_levels = sorted(zip(values, positions))[::-1]
    pos1 = sorted_levels[0][1]
    sorted_levels = filter(lambda w: abs(w[1] - pos1) > numpy.std(samples), sorted_levels)
    pos2 = sorted_levels[0][1]

    pos1, pos2 = min(pos1, pos2), max(pos1, pos2)
    threshold = ((pos2 - pos1)/2.0) + pos1
    print pos1, pos2, threshold

    plt.hist(samples, bins=100)
    plt.show()
    return samples

def stream_threshold(stream, threshold):
    for sample in stream:
        if sample > threshold:
            yield 1
        else:
            yield 0

def stream_pulse_widths(stream):
    sample_number = 0
    state, width = (0, 0)
    for sample in stream:
        sample_number += 1
        # 0 0
        if state == 0 and sample == 0:
            state, width = (0, 0)
        # 0 1
        elif state == 0 and sample == 1:
            offset = sample_number
            state, width = (1, 1)
        # 1 1
        elif state == 1 and sample == 1:
            state, width = (1, width+1)
        # 1 0
        elif state == 1 and sample == 0:
            yield (offset/float(SAMPLE_RATE), width/float(SAMPLE_RATE))
            state, width = (0, 0)

def stream_filter_pulse_widths(stream):
    state = []

    for (offset, width) in stream:
        if len(state) == 0:
            if width > 125e-3:
                state = [(offset, width)]
        else:
            # Emit the collected pulse if this one is 900ms past the beginning
            # of the collection
            if abs(offset - state[0][0]) > 900e-3:
                yield (state[0][0], sum([w for (_,w) in state]))
                state = [(offset, width)]
            # Add this pulse to our collection if it's less than 800ms from the
            # end of the last pulse in our collection
            elif abs(offset - state[-1][0]) < 800e-3:
                state.append((offset, width))

    # Flush the last pulse collection when input stream is terminated
    if len(state) > 0:
        yield (state[0][0], sum([w for (_,w) in state]))

def stream_pulse_widths_to_symbols(stream):
    # approximately equal means actual vs. +/-25% of expected
    approx_equal = lambda actual, expected: abs(actual - expected) < 0.25*expected

    for (offset, width) in stream:
        # 800ms for a marker
        if approx_equal(width, 800e-3):
            yield (offset, "M")
        # 500ms for a 1 bit
        elif approx_equal(width, 500e-3):
            yield (offset, 1)
        # 200ms for a 0 bit
        elif approx_equal(width, 200e-3):
            yield (offset, 0)
        # Invalid bit
        else:
            yield (offset, "I")

def stream_symbols_to_frame(stream):
    template = [     0 , 'B', 'B', 'B', 'B', 'B', 'B',  0 , 'M',
                    'B', 'B', 'B', 'B',  0 , 'B', 'B', 'B',  0 , 'M',
                    'B', 'B', 'B', 'B',  0 , 'B', 'B',  0 ,  0 , 'M',
                    'B', 'B', 'B', 'B',  0 , 'B', 'B', 'B', 'B', 'M',
                    'B', 'B',  0 ,  0 ,  0 ,  0 ,  0 , 'B',  0 , 'M', # FIXME
                    'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'M'    ]
    state = [None]*59

    for (offset, symbol) in stream:
        print (offset, symbol)
        state = state[1:] + [(offset, symbol)]
        if None not in state:
            # Check that the symbols are each 1.0s second +/-100ms apart
            offsets = [offset for (offset, _) in state]
            if not numpy.all(numpy.abs(numpy.diff(offsets) - 1.0) < 100e-3):
                continue

            # Check for no invalid symbols
            symbols = [symbol for (_, symbol) in state]
            if "I" in symbols:
                continue

            # Check that the symbols match the template
            compare = lambda expected, actual: (expected == 'B' and (actual == 0 or actual == 1)) or actual == expected
            matches = [ compare(template[i], state[i][1]) for i in range(len(template)) ]
            if not numpy.all(matches):
                continue

            # Emit the symbols of the valid frame
            print "Valid frame found!"
            yield symbols

            # Reset the state
            state = [None]*59

def stream_frame_to_wwv_record(stream):
    WWVRecord = collections.namedtuple('WWVRecord', ['DST1', 'LSW', 'Year', 'Minutes', 'Hours', 'Day_of_year', 'DUT1', 'DST2', 'UT1_Corr'])

    for frame in stream:
        dst1 = bool(frame[1])
        lsw = bool(frame[2])
        year = 1*frame[3] + 2*frame[4] + 4*frame[5] + 8*frame[6] + 10*frame[50] + 20*frame[51] + 40*frame[52] + 80*frame[53]
        minutes = 1*frame[9] + 2*frame[10] + 4*frame[11] + 8*frame[12] + 10*frame[14] + 20*frame[15] + 40*frame[16]
        hours = 1*frame[19] + 2*frame[20] + 4*frame[21] + 8*frame[22] + 10*frame[24] + 20*frame[25]
        day_of_year = 1*frame[29] + 2*frame[30] + 4*frame[31] + 8*frame[32] + 10*frame[34] + 20*frame[35] + 40*frame[36] + 80*frame[37] + 100*frame[39] + 200*frame[40]
        dut1 = bool(frame[49])
        dst2 = bool(frame[54])
        ut1_corr = 0.1*frame[55] + 0.2*frame[56] + 0.3*frame[57]

        yield WWVRecord(dst1, lsw, year, minutes, hours, day_of_year, dut1, dst2, ut1_corr)

def stream_print_wwv_record(stream):
    for record in stream:
        print record

def stream_plot(stream, n=None, title=""):
    if n is None:
        x = list(stream)
    else:
        x = [stream.next() for _ in range(n)]

    plt.plot(x)
    plt.ylabel('Value')
    plt.xlabel('Time (sample number)')
    plt.title(title)
    plt.show()

################################################################################

if len(sys.argv) < 2:
    print "Usage: %s <WWV recording wave file> [start] [stop]" % sys.argv[0]
    sys.exit(1)
elif len(sys.argv) == 2:
    s0 = stream_wave_file(sys.argv[1])
elif len(sys.argv) == 3:
    s0 = stream_wave_file(sys.argv[1], int(sys.argv[2]))
elif len(sys.argv) == 4:
    s0 = stream_wave_file(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

s1 = stream_bandpass_filter_iir(s0, 95.0, 105.0)
s2 = stream_rectify(s1)
s3 = stream_lowpass_filter_iir(s2, 5.0)
s4 = stream_threshold(s3, 1500)
s5 = stream_pulse_widths(s4)
s6 = stream_filter_pulse_widths(s5)
s7 = stream_pulse_widths_to_symbols(s6)
s8  = stream_symbols_to_frame(s7)
s9 = stream_frame_to_wwv_record(s8)
stream_print_wwv_record(s9)

