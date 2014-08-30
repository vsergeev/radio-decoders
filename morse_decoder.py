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
MORSE_FREQUENCY = None
MORSE_THRESHOLD = 5000.0
MORSE_DAH_MIN_LENGTH = 100e-3
MORSE_DIT_MAX_LENGTH = 90e-3

MORSE_GROUP_SEPARATION = 100e-3

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

@timed("Thresholding...")
def block_threshold(samples, threshold):
    samples = numpy.copy(samples)

    idx_above = samples > threshold
    idx_below = samples < threshold
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
        if abs(offset - last_offset) < MORSE_GROUP_SEPARATION:
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
                        '9': '----.',  '=': '-...-',   }

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
        if last_offset == None or (offset - last_offset) > 1.50:
            sys.stdout.write("\n%.2f:\t" % offset)
        elif (offset - last_offset) > 0.80:
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
samples = block_threshold(samples, MORSE_THRESHOLD)
samples = block_pulse_widths(samples)
samples = block_group_pulse_widths(samples)
samples = block_pulse_widths_to_symbols(samples)
samples = block_morse_to_ascii(samples)
block_ascii_to_conversation(samples)

