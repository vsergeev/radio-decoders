import sys
import scipy.io.wavfile
import scipy.signal
import numpy
import numpy.fft
import matplotlib.pyplot as plt
import collections

SAMPLE_RATE = None
RTTY_FREQUENCY = None
RTTY_THRESHOLD = None

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

######################################################################

def stream_wave_file(path, start=None, stop=None):
    global SAMPLE_RATE

    (SAMPLE_RATE, samples) = scipy.io.wavfile.read(path)

    if start is not None and stop is None:
        samples = samples[start*SAMPLE_RATE:]
    elif start is not None and stop is not None:
        samples = samples[start*SAMPLE_RATE:stop*SAMPLE_RATE]

    return samples

def stream_find_center_frequency(stream):
    global RTTY_FREQUENCY

    dft = numpy.abs(numpy.fft.rfft(stream))

    # Pick first strong frequency
    sorted_freqs = ((SAMPLE_RATE/2.0)/len(dft))*(numpy.argsort(dft)[::-1])
    freq1 = sorted_freqs[0]

    # Pick next strong frequeny at least 15 Hz away
    sorted_freqs = filter(lambda f: abs(f - freq1) > 15.0, sorted_freqs)
    freq2 = sorted_freqs[0]

    RTTY_FREQUENCY = tuple(sorted([freq1, freq2]))

    print "RTTY Frequency Pair: %.2f Hz / %.2f Hz" % RTTY_FREQUENCY
    print "Frequency shift: %.2f Hz" % abs(RTTY_FREQUENCY[0] - RTTY_FREQUENCY[1])

    return stream

def stream_bandpass_filter_iir_fast(stream, fLow, fHigh):
    b,a = scipy.signal.butter(5, [(2*fLow)/(SAMPLE_RATE), (2*fHigh)/(SAMPLE_RATE)], btype='bandpass')
    #plot_filter(b, a, SAMPLE_RATE)
    return scipy.signal.lfilter(b, a, stream)

def stream_zc_comparator(stream):
    state = 0

    for sample in stream:
        if state == 0 and sample > 0.0:
            state = 1
            yield 1.0
        elif state == 1 and sample < 0.0:
            state = 0
            yield 0.0
        elif state == 0 and sample < 0.0:
            state = 0
            yield 0.0
        elif state == 1 and sample > 0.0:
            state = 1
            yield 0.0

def stream_lowpass_filter_iir_fast(stream, fC):
    b, a = scipy.signal.butter(4, (2*fC)/SAMPLE_RATE)
    #plot_filter(b, a, SAMPLE_RATE, range(1000))
    return scipy.signal.lfilter(b, a, list(stream))

def stream_find_min_max(stream):
    global RTTY_THRESHOLD

    smax = numpy.percentile(stream, 85)
    smin = numpy.percentile(stream, 15)

    RTTY_THRESHOLD = ((smax - smin)/2.0) + smin

    return stream

def stream_rectify(stream):
    for sample in stream:
        yield abs(sample)

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

def stream_plot_dft(stream, title=""):
    samples_dft_mag = 20*numpy.log10(numpy.abs(numpy.fft.fft(stream)))
    samples_freqs = numpy.fft.fftfreq(len(samples_dft_mag), d=1.0/SAMPLE_RATE)
    plt.plot(samples_freqs, samples_dft_mag)
    plt.xlim([-9000, 9000])
    plt.ylabel('Amplitude (Log)')
    plt.xlabel('Frequency (Hz)')
    plt.title(title)
    plt.show()

if len(sys.argv) < 2:
    print "Usage: %s <WWV recording wave file> [start] [stop]" % sys.argv[0]
    sys.exit(1)
elif len(sys.argv) == 2:
    s0 = stream_wave_file(sys.argv[1])
elif len(sys.argv) == 3:
    s0 = stream_wave_file(sys.argv[1], int(sys.argv[2]))
elif len(sys.argv) == 4:
    s0 = stream_wave_file(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

s1 = stream_find_center_frequency(s0)
s2 = stream_bandpass_filter_iir_fast(s1, RTTY_FREQUENCY[0] - 50.0, RTTY_FREQUENCY[1] + 50.0)
s3 = stream_zc_comparator(s2)
s4 = stream_lowpass_filter_iir_fast(s3, 100.0)
s5 = stream_find_min_max(s4)
s6 = stream_threshold(s5, RTTY_THRESHOLD)
stream_plot(s6)

