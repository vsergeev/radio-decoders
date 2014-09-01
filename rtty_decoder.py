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

RTTY_LOW_MARK = True

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

samples = block_find_rtty_frequencies(samples)
samples = block_sliding_dft(samples)
samples = block_normalize(samples)
samples = block_threshold(samples)
block_plot(samples)

