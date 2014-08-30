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
            sys.stdout.write(" "*50 + "time: %.3f sec\n" % (toc-tic))
            return rets
        return wrapped_f
    return wrap

######################################################################

SAMPLE_RATE = None
THRESHOLD = None
MORSE_FREQUENCY = None
MORSE_DIT_MAX_LENGTH = None
MORSE_DAH_MIN_LENGTH = None

MORSE_LETTER_SEPARATION = 100e-3
MORSE_WORD_SEPARATION = 0.80
MORSE_SENTENCE_SEPARATION = 1.50

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

@timed("Finding center frequency...")
def block_find_center_frequency(samples):
    global MORSE_FREQUENCY

    dft = numpy.abs(numpy.fft.rfft(samples))
    peak_freq = ((SAMPLE_RATE/2.0)*numpy.argmax(dft))/len(dft)

    print "    CW Frequency: %.2f Hz" % peak_freq

    MORSE_FREQUENCY = peak_freq

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

@timed("Finding dit/dah threshold...")
def block_find_ditdah_threshold(samples):
    global MORSE_DAH_MIN_LENGTH
    global MORSE_DIT_MAX_LENGTH

    widths = numpy.array([w for (_, w) in samples])

    smin = numpy.min(widths)
    smax = numpy.max(widths)
    smean = numpy.mean(widths)
    stdev = numpy.std(widths)

    # Find two primary modes
    kernel = scipy.stats.gaussian_kde(widths)
    domain = numpy.linspace(smin, smax, 1000)
    estimate = kernel(domain)
    maxima, = scipy.signal.argrelmax(estimate)
    max1, max2 = sorted(zip(estimate[maxima], domain[maxima]))[::-1][0:2]
    smode1, smode2 = sorted((max1[1], max2[1]))

    threshold1 = ((smode2 - smode1)/3.0) + smode1
    threshold2 = (2.0*(smode2 - smode1)/3.0) + smode1

    print "    min %.2f  max %.2f  mean %.2f  stdev %.2f" % (smin, smax, smean, stdev)
    print "    mode1 %.2f  mode2 %.2f  threshold1 %.2f  threshold2 %.2f" % (smode1, smode2, threshold1, threshold2)

    MORSE_DIT_MAX_LENGTH = threshold1
    MORSE_DAH_MIN_LENGTH = threshold2

    #plt.plot(domain, kernel(domain))
    #plt.hist(widths, bins=50)
    #plt.axvline(threshold1, color='r', linestyle='dashed')
    #plt.axvline(threshold2, color='r', linestyle='dashed')
    #plt.show()

    return samples

@timed("Grouping pulse widths...")
def block_group_pulse_widths(samples):
    group_offset = None
    last_offset = None
    pulses = []

    for (offset, width) in samples:
        if group_offset == None or last_offset == None:
            group_offset = offset
            last_offset = offset + width

        # If this pulse width is still part of the same group
        if abs(offset - last_offset) < MORSE_LETTER_SEPARATION:
            pulses.append(width)
            last_offset = offset + width

        # Otherwise, emit the collected group and start a new one
        else:
            yield (group_offset, pulses)
            group_offset = offset
            last_offset = offset + width
            pulses = [width]

    yield (group_offset, pulses)

@timed("Converting pulse widths to symbols (dit/dahs)...")
def block_pulse_widths_to_symbols(samples):
    def pulse_width_to_morse_symbol(width):
        if width > MORSE_DAH_MIN_LENGTH:
            return "-"
        elif width < MORSE_DIT_MAX_LENGTH:
            return "."
        else:
            return "#"

    for (offset, pulses) in samples:
        morse = "".join(map(pulse_width_to_morse_symbol, pulses))
        yield (offset, morse)

@timed("Converting symbols to ASCII...")
def block_morse_to_ascii(samples):
    morse_table    = {  'A': '.-',     'B': '-...',   'C': '-.-.',
                        'D': '-..',    'E': '.',      'F': '..-.',
                        'G': '--.',    'H': '....',   'I': '..',
                        'J': '.---',   'K': '-.-',    'L': '.-..',
                        'M': '--',     'N': '-.',     'O': '---',
                        'P': '.--.',   'Q': '--.-',   'R': '.-.',
                        'S': '...',    'T': '-',      'U': '..-',
                        'V': '...-',   'W': '.--',    'X': '-..-',
                        'Y': '-.--',   'Z': '--..',
                        '0': '-----',  '1': '.----',  '2': '..---',
                        '3': '...--',  '4': '....-',  '5': '.....',
                        '6': '-....',  '7': '--...',  '8': '---..',
                        '9': '----.',  '=': '-...-',   
                        '[BK]': '-...-.-',
                        '[SN]': '...-.',
                    }

    # Invert table to map morse to letter
    morse_table = {v: k for k, v in morse_table.items()}

    for (offset, morse) in samples:
        if morse in morse_table:
            yield (offset, morse_table[morse])
        else:
            yield (offset, morse)

@timed("Converting ASCII to conversation...")
def block_ascii_to_conversation(samples):
    last_offset = None

    for (offset, letter) in samples:
        if last_offset == None or (offset - last_offset) > MORSE_SENTENCE_SEPARATION:
            sys.stdout.write("\n%.2f:\t" % offset)
        elif (offset - last_offset) > MORSE_WORD_SEPARATION:
            sys.stdout.write(" ")

        sys.stdout.write(letter)

        last_offset = offset

    sys.stdout.write("\n")

def block_plot(samples, n=None, title=""):
    if n is None:
        x = list(samples)
    else:
        x = [samples.next() for _ in range(n)]

    plt.plot(x)
    plt.ylabel('Value')
    plt.xlabel('Time (sample number)')
    plt.title(title)
    plt.show()

################################################################################

if len(sys.argv) < 2:
    print "Usage: %s <recorded Morse wave file> [start] [stop]" % sys.argv[0]
    sys.exit(1)
elif len(sys.argv) == 2:
    samples = block_wave_file(sys.argv[1])
elif len(sys.argv) == 3:
    samples = block_wave_file(sys.argv[1], int(sys.argv[2]))
elif len(sys.argv) == 4:
    samples = block_wave_file(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

samples = block_find_center_frequency(samples)
samples = block_bandpass_filter_iir(samples, MORSE_FREQUENCY - 10.0, MORSE_FREQUENCY + 10.0)
samples = block_rectify(samples)
samples = block_lowpass_filter_iir(samples, 50.0)
samples = block_find_threshold(samples)
samples = block_threshold(samples)
samples = block_pulse_widths(samples)
samples = block_find_ditdah_threshold(samples)
samples = block_group_pulse_widths(samples)
samples = block_pulse_widths_to_symbols(samples)
samples = block_morse_to_ascii(samples)
block_ascii_to_conversation(samples)

