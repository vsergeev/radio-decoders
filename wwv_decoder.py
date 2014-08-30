import sys
import time
import collections
import numpy
import numpy.fft
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt

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

def timed(message):
    def wrap(f):
        def wrapped_f(*args, **kwargs):
            sys.stdout.write(message + "\n")
            sys.stdout.flush()
            tic = time.time()
            rets = f(*args, **kwargs)
            toc = time.time()
            sys.stdout.write(" "*50 + "%.3f sec\n" % (toc-tic))
            return rets
        return wrapped_f
    return wrap

################################################################################

SAMPLE_RATE = None
THRESHOLD = None

################################################################################

@timed("Reading wave file...")
def block_wave_file(path, start=None, stop=None):
    global SAMPLE_RATE

    (SAMPLE_RATE, samples) = scipy.io.wavfile.read(path, mmap=True)

    if start is not None and stop is None:
        samples = samples[start*SAMPLE_RATE:]
    elif start is not None and stop is not None:
        samples = samples[start*SAMPLE_RATE:stop*SAMPLE_RATE]

    return samples

@timed("Bandpass filtering...")
def block_bandpass_filter_iir(samples, fLow, fHigh):
    b,a = scipy.signal.butter(3, [(2*fLow)/(SAMPLE_RATE), (2*fHigh)/(SAMPLE_RATE)], btype='bandpass')
    #plot_filter(b, a, SAMPLE_RATE, range(1000))
    return scipy.signal.lfilter(b, a, samples)

@timed("Rectifying...")
def block_rectify(samples):
    return numpy.abs(samples)

@timed("Low pass filtering...")
def block_lowpass_filter_iir(samples, fC):
    b, a = scipy.signal.butter(4, (2*fC)/SAMPLE_RATE)
    #plot_filter(b, a, SAMPLE_RATE, range(1000))
    return scipy.signal.lfilter(b, a, samples)

@timed("Finding threshold...")
def block_find_threshold(samples):
    global THRESHOLD

    counts, bins = numpy.histogram(samples, bins=100)
    smode = bins[numpy.argmax(counts)]

    smin = numpy.min(samples)
    smax = numpy.max(samples)
    smean = numpy.mean(samples)
    stdev = numpy.std(samples)

    THRESHOLD = smode + stdev

    print "    min %.2f  max %.2f  mean %.2f  stdev %.2f" % (smin, smax, smean, stdev)
    print "    approx. mode %.2f  threshold %.2f" % (smode, THRESHOLD)

    #plt.hist(samples, bins=100)
    #plt.axvline(THRESHOLD, color='r', linestyle='dashed')
    #plt.show()

    return samples

@timed("Thresholding...")
def block_threshold(samples):
    samples = numpy.copy(samples)

    idx_above = samples > THRESHOLD
    idx_below = samples < THRESHOLD
    samples[idx_above] = 1
    samples[idx_below] = 0

    return samples

@timed("Converting samples to pulse widths...")
def block_pulse_widths(samples):
    widths = []

    markers = numpy.diff(samples)
    starts, = numpy.where(markers > 0)
    stops, = numpy.where(markers < 0)
    widths = (stops - starts)/float(SAMPLE_RATE)
    offsets = (starts + 1)/float(SAMPLE_RATE)

    pulses = zip(offsets, widths)

    return pulses

@timed("Filtering pulse widths...")
def block_filter_pulse_widths(samples):
    state = []

    for (offset, width) in samples:
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

    # Flush the last pulse collection when input samples is terminated
    if len(state) > 0:
        yield (state[0][0], sum([w for (_,w) in state]))

@timed("Converting pulse widths to symbols...")
def block_pulse_widths_to_symbols(samples):
    # approximately equal means actual is within +/- 25% of expected
    approx_equal = lambda actual, expected: abs(actual - expected) < 0.25*expected

    for (offset, width) in samples:
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

@timed("Converting symbols to frame...")
def block_symbols_to_frame(samples):
    template = [     0 , 'B', 'B', 'B', 'B', 'B', 'B',  0 , 'M',
                    'B', 'B', 'B', 'B',  0 , 'B', 'B', 'B',  0 , 'M',
                    'B', 'B', 'B', 'B',  0 , 'B', 'B',  0 ,  0 , 'M',
                    'B', 'B', 'B', 'B',  0 , 'B', 'B', 'B', 'B', 'M',
                    'B', 'B',  0 ,  0 ,  0 ,  0 ,  0 , 'B',  0 , 'M', # FIXME
                    'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'M'    ]
    state = [None]*59

    for (offset, symbol) in samples:
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

def block_frame_to_wwv_record(samples):
    WWVRecord = collections.namedtuple('WWVRecord', ['DST1', 'LSW', 'Year', 'Minutes', 'Hours', 'Day_of_year', 'DUT1', 'DST2', 'UT1_Corr'])

    for frame in samples:
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

def block_print_wwv_record(samples):
    for record in samples:
        print record

def block_plot(samples, n=None, title=""):
    plt.plot(samples[0:n])
    plt.ylabel('Value')
    plt.xlabel('Time (sample number)')
    plt.title(title)
    plt.show()

################################################################################

if len(sys.argv) < 2:
    print "Usage: %s <recorded WWV wave file> [start] [stop]" % sys.argv[0]
    sys.exit(1)
elif len(sys.argv) == 2:
    samples = block_wave_file(sys.argv[1])
elif len(sys.argv) == 3:
    samples = block_wave_file(sys.argv[1], int(sys.argv[2]))
elif len(sys.argv) >= 4:
    samples = block_wave_file(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

samples = block_bandpass_filter_iir(samples, 95.0, 105.0)
samples = block_rectify(samples)
samples = block_lowpass_filter_iir(samples, 5.0)
samples = block_find_threshold(samples)
samples = block_threshold(samples)
samples = block_pulse_widths(samples)
samples = block_filter_pulse_widths(samples)
samples = block_pulse_widths_to_symbols(samples)
samples  = block_symbols_to_frame(samples)
samples = block_frame_to_wwv_record(samples)
block_print_wwv_record(samples)

