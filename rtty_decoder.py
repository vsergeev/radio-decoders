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

    dft = numpy.abs(numpy.fft.rfft(samples))

    # Pick first strong frequency
    sorted_freqs = ((SAMPLE_RATE/2.0)/len(dft))*(numpy.argsort(dft)[::-1])
    freq1 = sorted_freqs[0]

    # Pick next strong frequeny at least 15 Hz away
    sorted_freqs = filter(lambda f: abs(f - freq1) > 15.0, sorted_freqs)
    freq2 = sorted_freqs[0]

    RTTY_FREQUENCY = tuple(sorted([freq1, freq2]))

    print "RTTY Frequency Pair: %.2f Hz / %.2f Hz" % RTTY_FREQUENCY
    print "Frequency shift: %.2f Hz" % abs(RTTY_FREQUENCY[0] - RTTY_FREQUENCY[1])

    return samples 

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

    return samples

@timed("Thresholding...")
def block_threshold(samples, threshold):
    samples = numpy.copy(samples)

    idx_above = samples > threshold
    idx_below = samples < threshold
    samples[idx_above] = 1
    samples[idx_below] = 0

    return samples

def block_plot(samples, n=None, title=""):
    plt.plot(samples[0:n])
    plt.ylabel('Value')
    plt.xlabel('Time (sample number)')
    plt.title(title)
    plt.show()

def block_plot_dft(samples, title=""):
    samples_dft_mag = 20*numpy.log10(numpy.abs(numpy.fft.fft(samples)))
    samples_freqs = numpy.fft.fftfreq(len(samples_dft_mag), d=1.0/SAMPLE_RATE)
    plt.plot(samples_freqs, samples_dft_mag)
    plt.xlim([-9000, 9000])
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

samples = block_find_rtty_frequencies(samples)
samples = block_bandpass_filter_iir(samples, RTTY_FREQUENCY[0] - 50.0, RTTY_FREQUENCY[1] + 50.0)
samples = block_zc_comparator(samples)
samples = block_lowpass_filter_iir(samples, 100.0)
samples = block_find_threshold(samples)
samples = block_threshold(samples, RTTY_THRESHOLD)
block_plot(samples)

