import sys
import time
import collections
import numpy
import numpy.fft
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt

######################################################################

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

######################################################################

SAMPLE_RATE = None
RTTY_FREQUENCY = None
RTTY_THRESHOLD = None

RTTY_LOW_MARK = False
RTTY_BAUDRATE = 45.45
RTTY_START_BITS = 1
RTTY_DATA_BITS = 5
RTTY_STOP_BITS = 1.5
RTTY_LSB_FIRST = True

######################################################################

@timed("Reading wave file...")
def block_wave_file(path, start=None, stop=None):
    global SAMPLE_RATE

    (SAMPLE_RATE, samples) = scipy.io.wavfile.read(path, mmap=True)

    if start is not None and stop is None:
        samples = samples[start*SAMPLE_RATE:]
    elif start is not None and stop is not None:
        samples = samples[start*SAMPLE_RATE:stop*SAMPLE_RATE]

    return samples

@timed("Finding RTTY frequencies...")
def block_find_rtty_frequencies(samples):
    global RTTY_FREQUENCY

    magdft = numpy.abs(numpy.fft.rfft(samples))

    # Pick first strong frequency
    sorted_freqs = ((SAMPLE_RATE/2.0)/len(magdft))*(numpy.argsort(magdft)[::-1])
    freq1 = sorted_freqs[0]

    # Pick next strong frequeny at least 15 Hz away
    sorted_freqs = filter(lambda f: abs(f - freq1) > 15.0, sorted_freqs)
    freq2 = sorted_freqs[0]

    RTTY_FREQUENCY = tuple(sorted([freq1, freq2]))

    print "    RTTY Frequency Pair: %.2f Hz / %.2f Hz" % RTTY_FREQUENCY
    print "    Frequency shift: %.2f Hz" % abs(RTTY_FREQUENCY[0] - RTTY_FREQUENCY[1])

    return samples

@timed("Reading wave file...")
def block_wave_file(path, start=None, stop=None):
    global SAMPLE_RATE

    (SAMPLE_RATE, samples) = scipy.io.wavfile.read(path, mmap=True)

    if start is not None and stop is None:
        samples = samples[start*SAMPLE_RATE:]
    elif start is not None and stop is not None:
        samples = samples[start*SAMPLE_RATE:stop*SAMPLE_RATE]

    print "    sample rate %d" % SAMPLE_RATE

    return samples

@timed("Performing sliding DFT...")
def block_sliding_dft(samples):
    rtty_delta = RTTY_FREQUENCY[1] - RTTY_FREQUENCY[0]
    N = int(SAMPLE_RATE/(rtty_delta/5.0))
    rtty_low_index = int((RTTY_FREQUENCY[0]/(SAMPLE_RATE/2.0)) * N/2.0)
    rtty_high_index = int((RTTY_FREQUENCY[1]/(SAMPLE_RATE/2.0)) * N/2.0)

    print "    N %d  low index %d  high index %d" % (N, rtty_low_index, rtty_high_index)

    wf = numpy.hanning(N)

    spectrogram = []
    for n in range(len(samples) - N - (len(samples) % N)):
        sample_window = numpy.array(samples[n:n+N])*wf
        dft = numpy.fft.rfft(sample_window)
        spectrogram.append((abs(dft[rtty_low_index]), abs(dft[rtty_high_index])))

    #pspectrogram = numpy.abs(numpy.array(spectrogram).T)
    #plt.imshow(pspectrogram, origin='lower', aspect='auto')
    #plt.show()

    return spectrogram

@timed("Normalizing...")
def block_normalize(samples):
    samples = numpy.array(samples).T

    #plt.plot(samples[0], label="low")
    #plt.plot(samples[1], label="high")
    #plt.legend()
    #plt.show()

    samples[0] = samples[0]*(numpy.mean(samples[1])/numpy.mean(samples[0]))
    return zip(samples[0], samples[1])

@timed("Thresholding...")
def block_threshold(samples):
    nsamples = []
    for sample in samples:
        nsamples.append(1*(sample[1] > sample[0]) ^ RTTY_LOW_MARK)

    #plt.plot(nsamples)
    #plt.show()

    return nsamples

@timed("Bandpass filtering...")
def block_bandpass_filter_iir(samples, fLow, fHigh):
    b,a = scipy.signal.butter(5, [(2*fLow)/(SAMPLE_RATE), (2*fHigh)/(SAMPLE_RATE)], btype='bandpass')
    #plot_filter(b, a, SAMPLE_RATE, range(1000))
    return scipy.signal.lfilter(b, a, samples)

def block_zc_comparator(samples):
    nsamples = []
    state = 0

    for sample in samples:
        if state == 0 and sample > 0.0:
            state = 1
            nsamples.append(1.0)
        elif state == 1 and sample < 0.0:
            state = 0
            nsamples.append(0.0)
        elif state == 0 and sample < 0.0:
            state = 0
            nsamples.append(0.0)
        elif state == 1 and sample > 0.0:
            state = 1
            nsamples.append(0.0)

    return nsamples

@timed("Low pass filtering...")
def block_lowpass_filter_iir(samples, fC):
    b, a = scipy.signal.butter(4, (2*fC)/SAMPLE_RATE)
    #plot_filter(b, a, SAMPLE_RATE, range(1000))
    return scipy.signal.lfilter(b, a, samples)

@timed("Rectifying...")
def block_rectify(samples):
    return numpy.abs(samples)

@timed("Finding threshold...")
def block_find_threshold(samples):
    global RTTY_THRESHOLD

    smax = numpy.percentile(samples, 85)
    smin = numpy.percentile(samples, 15)

    RTTY_THRESHOLD = ((smax - smin)/2.0) + smin

    #plt.plot(numpy.arange(len(samples))/float(SAMPLE_RATE), samples)
    #plt.axhline(RTTY_THRESHOLD)
    #plt.show()

    return samples

@timed("Thresholding...")
def block_threshold2(samples, threshold):
    samples = numpy.copy(samples)

    idx_above = samples > threshold
    idx_below = samples < threshold
    samples[idx_above] = 1 ^ RTTY_LOW_MARK
    samples[idx_below] = 0 ^ RTTY_LOW_MARK

    return samples

@timed("Decoding bits...")
def block_decode(samples):
    bits_per_frame = RTTY_START_BITS + RTTY_DATA_BITS + RTTY_STOP_BITS

    oversample = 2
    sample_offsets = (numpy.arange(1/(2*oversample*RTTY_BAUDRATE), bits_per_frame/RTTY_BAUDRATE, 1/(oversample*RTTY_BAUDRATE))*SAMPLE_RATE).astype(int)

    sample_window_len = sample_offsets[-1]+1
    sample_window = [1]*sample_window_len

    #bit_offsets = (numpy.arange(0, (bits_per_frame+1)/RTTY_BAUDRATE, 1/RTTY_BAUDRATE)*SAMPLE_RATE).astype(int)
    #x = numpy.arange(0, int(SAMPLE_RATE*(8/RTTY_BAUDRATE)))
    #plt.plot(x, [0.0]*len(x))
    #for z in sample_offsets:
    #    plt.axvline(z, color='b')
    #for z in bit_offsets:
    #    plt.axvline(z, color='r')
    #plt.show()

    bit_consensus = lambda x: (numpy.all(x == 1) or numpy.all(x == 0))
    #bit_consensus = lambda x: (len(x[x == 1]) > (3*oversample/4) or len(x[x == 0]) > (3*oversample/4))
    #bit_majority = lambda x: (len(x[x == 1]) > len(x[x == 0]))*1

    nsamples = []
    sample_number = 0
    counter = 0

    for sample in samples:
        sample_window = sample_window[1:] + [sample]
        sample_number += 1
        counter += 1

        if sample_window[0] == 1 and sample_window[1] == 0:
            # Sample bits at sample offsets
            bit_samples = numpy.array(sample_window)[sample_offsets]

            # Isolate start, data, stop bit samples
            start_samples = bit_samples[0:oversample]
            data_samples = [bit_samples[oversample*i:oversample*(i+1)] for i in range(1, 1+RTTY_DATA_BITS)]
            stop_samples = bit_samples[(RTTY_START_BITS+RTTY_DATA_BITS)*oversample:]

            # Verify bit consensus, start bit and stop bit values
            if not bit_consensus(start_samples) or start_samples[0] != 0:
                print "Start bit failure"
                continue
            if not bit_consensus(stop_samples) or stop_samples[0] != 1:
                print "Stop bit failure"
                continue
            if not numpy.all([bit_consensus(d) for d in data_samples]):
                print "Data bits failure"
                continue

            offset = sample_number - sample_window_len
            print "%d: Found frame" % (offset)

            # Extract the data bits
            data = [int(d[0]) for d in data_samples]
            nsamples.append(data)

            # Reset the sample window
            sample_window = [1]*sample_window_len
            counter = 0

    return nsamples

@timed("Converting bits to characters...")
def block_bits_to_ita2(samples):
    ita2_table = {  False: { 0b00000: '[0]', 0b00100: ' ' , 0b10111: 'Q',
                             0b10011: 'W', 0b00001: 'E', 0b01010: 'R',
                             0b10000: 'T', 0b10101: 'Y', 0b00111: 'U',
                             0b00110: 'I', 0b11000: 'O', 0b10110: 'P',
                             0b00011: 'A', 0b00101: 'S', 0b01001: 'D',
                             0b01101: 'F', 0b11010: 'G', 0b10100: 'H',
                             0b01011: 'J', 0b01111: 'K', 0b10010: 'L',
                             0b10001: 'Z', 0b11101: 'X', 0b01110: 'C',
                             0b11110: 'V', 0b11001: 'B', 0b01100: 'N',
                             0b11100: 'M', 0b01000: '\r', 0b00010: '\n', },
                    True: {  0b00000: '[0]', 0b00100: ' ', 0b10111: '1',
                             0b10011: '2', 0b00001: '3', 0b01010: '4',
                             0b10000: '5', 0b10101: '6', 0b00111: '7',
                             0b00110: '8', 0b11000: '9', 0b10110: '0',
                             0b00011: '-', 0b00101: '[Bell]', 0b01001: '[WRU?]',
                             0b01101: '!', 0b11010: '&', 0b10100: '#',
                             0b01011: '\'', 0b01111: '(', 0b10010: ')',
                             0b10001: '"', 0b11101: '/', 0b01110: ':',
                             0b11110: ';', 0b11001: '?', 0b01100: ',',
                             0b11100: '.', 0b01000: '\r', 0b00010: '\n', } }
    state_shifted = False

    for sample in samples:
        if sample is None:
            yield "`"
            continue

        if RTTY_LSB_FIRST:
            sample = sample[::-1]

        bits = (sample[0] << 4) | (sample[1] << 3) | (sample[2] << 2) | (sample[3] << 1) | (sample[4] << 0)
        if bits == 0b11011:
            state_shifted = True
        elif bits == 0b11111:
            state_shifted = False
        else:
            yield ita2_table[state_shifted][bits]

@timed("Printing conversation...")
def block_print_conversation(samples):
    for sample in samples:
        sys.stdout.write(sample)

def block_plot(samples, n=None, title=""):
    plt.plot(numpy.arange(len(samples[0:n]))/float(SAMPLE_RATE), samples[0:n])
    plt.ylabel('Value')
    plt.xlabel('Time (seconds)')
    plt.title(title)
    plt.show()

def block_plot_dft(samples, title=""):
    samples_dft_mag = 20*numpy.log10(numpy.abs(numpy.fft.fft(samples)))
    samples_freqs = numpy.fft.fftfreq(len(samples_dft_mag), d=1.0/SAMPLE_RATE)
    plt.plot(samples_freqs, samples_dft_mag)
    #plt.xlim([-9000, 9000])
    plt.ylabel('Amplitude (Log)')
    plt.xlabel('Frequency (Hz)')
    plt.title(title)
    plt.show()

if len(sys.argv) < 2:
    print "Usage: %s <recorded RTTY wave file> [start] [stop]" % sys.argv[0]
    sys.exit(1)
elif len(sys.argv) == 2:
    samples = block_wave_file(sys.argv[1])
elif len(sys.argv) == 3:
    samples = block_wave_file(sys.argv[1], int(sys.argv[2]))
elif len(sys.argv) == 4:
    samples = block_wave_file(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

######################################################################

samples = block_find_rtty_frequencies(samples)
samples = block_bandpass_filter_iir(samples, RTTY_FREQUENCY[0] - 50.0, RTTY_FREQUENCY[1] + 50.0)
samples = block_zc_comparator(samples)
samples = block_lowpass_filter_iir(samples, 125.0)
samples = block_find_threshold(samples)
samples = block_threshold2(samples, RTTY_THRESHOLD)
samples = block_decode(samples)
samples = block_bits_to_ita2(samples)
block_print_conversation(samples)

