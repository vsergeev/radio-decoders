import sys
import scipy.io.wavfile
import scipy.signal
import scipy.stats
import numpy
import numpy.fft
import matplotlib.pyplot as plt
import collections

MORSE_THRESHOLD = 5000.0

SAMPLE_RATE = None
MORSE_FREQUENCY = None
MORSE_DAH_MIN_LENGTH = 100e-3
MORSE_DIT_MAX_LENGTH = 90e-3
MORSE_GROUP_SEPARATION = 100e-3

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

    (SAMPLE_RATE, samples) = scipy.io.wavfile.read(path, mmap=True)

    if start is not None and stop is None:
        samples = samples[start*SAMPLE_RATE:]
    elif start is not None and stop is not None:
        samples = samples[start*SAMPLE_RATE:stop*SAMPLE_RATE]

    return samples

def stream_find_center_frequency(stream):
    global MORSE_FREQUENCY

    dft = numpy.abs(numpy.fft.rfft(stream))
    peak_freq = ((SAMPLE_RATE/2.0)*numpy.argmax(dft))/len(dft)

    print "CW Frequency: %.2f Hz" % peak_freq

    MORSE_FREQUENCY = peak_freq

    return stream

def stream_bandpass_filter_iir_fast(stream, fLow, fHigh):
    b,a = scipy.signal.butter(3, [(2*fLow)/(SAMPLE_RATE), (2*fHigh)/(SAMPLE_RATE)], btype='bandpass')
    #plot_filter(b, a, SAMPLE_RATE, range(1000))
    return scipy.signal.lfilter(b, a, stream)

def stream_rectify(stream):
    for sample in stream:
        yield abs(sample)

def stream_lowpass_filter_iir_fast(stream, fC):
    b, a = scipy.signal.butter(4, (2*fC)/SAMPLE_RATE)
    #plot_filter(b, a, SAMPLE_RATE, range(1000))
    return scipy.signal.lfilter(b, a, list(stream))

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

def stream_pulse_widths_hist(stream):
    global MORSE_DAH_MIN_LENGTH
    global MORSE_DIT_MAX_LENGTH

    pulse_widths = list(stream)

    widths = numpy.array([w for (_, w) in pulse_widths])

    values, positions = numpy.histogram(widths, bins=50)
    sorted_widths = sorted(zip(values, positions))[::-1]
    pos1 = sorted_widths[0][1]
    sorted_widths = filter(lambda w: abs(w[1] - pos1) > numpy.std(widths), sorted_widths)
    pos2 = sorted_widths[0][1]

    pos1, pos2 = min(pos1, pos2), max(pos1, pos2)

    threshold = ((pos2 - pos1)/2.0) + pos1
    MORSE_DAH_MIN_LENGTH = threshold
    MORSE_DAX_MIN_LENGTH = threshold

    plt.hist(widths, bins=50)
    plt.show()
    return pulse_widths

def stream_group_pulse_widths(stream):
    group_offset = None
    last_offset = None
    pulses = []

    for (offset, width) in stream:
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

def stream_pulse_widths_to_symbols(stream):
    #approx_equal = lambda actual, expected: abs(actual - expected) < 0.30*expected

    def pulse_width_to_morse_symbol(width):
        if width > MORSE_DAH_MIN_LENGTH:
            return "-"
        elif width < MORSE_DIT_MAX_LENGTH:
            return "."
        else:
            return "#"

    for (offset, pulses) in stream:
        morse = "".join(map(pulse_width_to_morse_symbol, pulses))
        yield (offset, morse)

def stream_morse_to_ascii(stream):
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

    for (offset, morse) in stream:
        if morse in morse_table:
            yield (offset, morse_table[morse])
        else:
            yield (offset, morse)

def stream_ascii_to_conversation(stream):
    last_offset = None

    for (offset, letter) in stream:
        if last_offset == None or (offset - last_offset) > 1.50:
            sys.stdout.write("\n%.2f:\t" % offset)
        elif (offset - last_offset) > 0.50:
            sys.stdout.write(" ")

        sys.stdout.write(letter)

        last_offset = offset

    sys.stdout.write("\n")

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
s2 = stream_bandpass_filter_iir_fast(s1, MORSE_FREQUENCY - 10.0, MORSE_FREQUENCY + 10.0)
s3 = stream_rectify(s2)
s4 = stream_lowpass_filter_iir_fast(s3, 50.0)
s5 = stream_threshold(s4, MORSE_THRESHOLD)
s6 = stream_pulse_widths(s5)
s7 = stream_group_pulse_widths(s6)
s8 = stream_pulse_widths_to_symbols(s7)
s9 = stream_morse_to_ascii(s8)
stream_ascii_to_conversation(s9)

