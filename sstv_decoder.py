import sys
import time
import collections
import enum
import numpy
import numpy.fft
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
import PIL

######################################################################

SAMPLE_RATE = None

######################################################################

def block_wave_file(path, start=None, stop=None):
    global SAMPLE_RATE

    (SAMPLE_RATE, samples) = scipy.io.wavfile.read(path, mmap=True)

    if start is not None and stop is None:
        samples = samples[int(start*SAMPLE_RATE):]
    elif start is not None and stop is not None:
        samples = samples[int(start*SAMPLE_RATE):int(stop*SAMPLE_RATE)]

    return samples

def block_differentiate(samples):
    return scipy.signal.lfilter([-1/12, 2/3, 0, -2/3, 1/12], 1, samples)

def block_rectify(samples):
    return numpy.abs(samples)

def block_lowpass_filter_fir(samples, fC):
    b = scipy.signal.firwin(1024, (2*fC)/SAMPLE_RATE)
    #plot_filter(b, 1, SAMPLE_RATE, range(1000))
    return scipy.signal.lfilter(b, 1, samples)

def stream_samples_to_stream(samples):
    for sample in samples:
        yield sample

def decode_sstv(stream, width=320, bits=8):
    class SSTVDecoderState(enum.Enum):
        CALIBRATION = 1
        VIS = 2
        SYNC = 3
        SCANLINE_B = 4
        SCANLINE_G = 5
        SCANLINE_R = 6

    state = SSTVDecoderState.CALIBRATION
    sample_offset = 0

    # Calibration sampling state
    sample_window = []
    frequency_slope = None

    # VIS sampling state
    vis_bit_sample_offset = None
    vis_bits = []

    # Sync sampling state
    sync_sample_offset = None

    # Scanline sampling state
    scanline_sample_offset = None
    scanline_count = 0

    # Outputs
    vis_mode = None
    image_r = []
    image_g = []
    image_b = []

    # Constants
    calibration_window_length = int((635/1000)*SAMPLE_RATE)
    color_sampling_offset = int(((146.432 / 1000)/width)*SAMPLE_RATE)

    assert color_sampling_offset > 1, "Insufficient sample rate for this resolution!"

    for sample in stream:
        sample_offset += 1

        if state == SSTVDecoderState.CALIBRATION:
            # Scan for 300ms Leader, 30ms break, 300ms Leader in sample window

            # Collect 635ms sample window
            if len(sample_window) < calibration_window_length:
                sample_window.append(sample)
                continue
            else:
                sample_window = sample_window[1:] + [sample]

            # Calculate mean of samples
            # FIXME max
            dc = (max(sample_window) + (max(sample_window)/1900)*1200)/2

            # Threshold samples about mean
            sample_window_copy = numpy.array(sample_window)
            threshold_samples = numpy.array(sample_window)
            threshold_samples[sample_window_copy > dc] = 1
            threshold_samples[sample_window_copy < dc] = 0

            # Calculate edges of pulses
            markers = numpy.diff(threshold_samples)
            starts, = numpy.where(markers > 0)
            stops, = numpy.where(markers < 0)

            if len(starts) == 2 and len(stops) == 2:
                # Check for approx. 300ms, 30ms, 300ms
                if abs((stops[0] - starts[0])/SAMPLE_RATE - 0.300) < 0.10*0.300 and \
                   abs((starts[1] - stops[0])/SAMPLE_RATE - 0.030) < 0.50*0.030 and \
                   abs((stops[1] - starts[1])/SAMPLE_RATE - 0.300) < 0.10*0.300:
                    # Found calibration pattern

                    # Sample frequency relationship
                    frequency_slope = 1900/numpy.mean(sample_window[starts[0]+1:stops[0]-1])

                    # Calculate first VIS bit offset
                    # Skip the start bit, and move to middle of first data bit
                    vis_bit_sample_offset = (sample_offset - calibration_window_length) + stops[1] + int((0.030 + 0.030/2)*SAMPLE_RATE)

                    state = SSTVDecoderState.VIS

        elif state == SSTVDecoderState.VIS:
            # Wait until we're at the sync pulse offset
            if sample_offset != vis_bit_sample_offset:
                continue

            # Slice at 1200 Hz, below (1100 Hz) is a 1, above (1300 Hz) is a 0
            vis_bits.append(int(sample < 1200/frequency_slope))

            if len(vis_bits) == 8:
                # Convert vis bits to integer
                vis_mode = 0
                for bit in vis_bits[:7][::-1]:
                    vis_mode = (vis_mode << 1) | bit

                # Check parity
                if (vis_bits.count(1) % 2) != 0:
                    print("Error: VIS mode bits parity mismatch.")
                else:
                    print("VIS Mode: {}".format(vis_mode))

                sync_sample_offset = vis_bit_sample_offset + int((0.030/2) * SAMPLE_RATE)
                state = SSTVDecoderState.SYNC
            else:
                # Calculate offset to sample next bit
                vis_bit_sample_offset += int(0.030*SAMPLE_RATE)

        elif state == SSTVDecoderState.SYNC:
            # Wait until we're at the sync pulse offset
            if sample_offset < sync_sample_offset:
                continue

            # Check if we're at the sync pulse (below 1500 Hz)
            if sample < 1500/frequency_slope:
                # Calculate the approximate next sync offset
                sync_sample_offset += int((114.3/256)*SAMPLE_RATE)

                scanline_count += 1

                if scanline_count <= 256:
                    image_r.append([])
                    image_g.append([])
                    image_b.append([])
                    scanline_sample_offset = sample_offset + int(((4.862+0.572)/1000)*SAMPLE_RATE) + color_sampling_offset // 2
                    state = SSTVDecoderState.SCANLINE_G
                else:
                    break

        elif state == SSTVDecoderState.SCANLINE_G:
            # Wait until we're at the scanline sample offset
            if sample_offset < scanline_sample_offset:
                continue

            # Compute pixel
            pixel = max(int((sample - 1500/frequency_slope)*(2**bits/(2300/frequency_slope - 1500/frequency_slope))), 0)

            # Append it to this scanline in the image channel
            image_g[scanline_count - 1].append(pixel)

            if len(image_g[scanline_count - 1]) < width:
                scanline_sample_offset += color_sampling_offset
            else:
                scanline_sample_offset += int(((0.572)/1000)*SAMPLE_RATE) + color_sampling_offset
                state = SSTVDecoderState.SCANLINE_B

        elif state == SSTVDecoderState.SCANLINE_B:
            # Wait until we're at the scanline sample offset
            if sample_offset < scanline_sample_offset:
                continue

            # Compute pixel
            pixel = max(int((sample - 1500/frequency_slope)*(2**bits/(2300/frequency_slope - 1500/frequency_slope))), 0)

            # Append it to this scanline in the image channel
            image_b[scanline_count - 1].append(pixel)

            if len(image_b[scanline_count - 1]) < width:
                scanline_sample_offset += color_sampling_offset
            else:
                scanline_sample_offset += int(((0.572)/1000)*SAMPLE_RATE) + color_sampling_offset
                state = SSTVDecoderState.SCANLINE_R

        elif state == SSTVDecoderState.SCANLINE_R:
            # Wait until we're at the scanline sample offset
            if sample_offset < scanline_sample_offset:
                continue

            # Compute pixel
            pixel = max(int((sample - 1500/frequency_slope)*(2**bits/(2300/frequency_slope - 1500/frequency_slope))), 0)

            # Append it to this scanline in the image channel
            image_r[scanline_count - 1].append(pixel)

            if len(image_r[scanline_count - 1]) < width:
                scanline_sample_offset += color_sampling_offset
            else:
                state = SSTVDecoderState.SYNC

    # Convert rgb scanline channels into an image
    im = PIL.Image.new("RGB", (width, 256))
    pixels = im.load()
    for y in range(256):
        for x in range(width):
            pixels[x,y] = (image_r[y][x], image_g[y][x], image_b[y][x])

    return im

######################################################################

def block_plot(samples, n=None, title=""):
    plt.plot(samples[0:n])
    plt.ylabel('Value')
    plt.xlabel('Time (sample number)')
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

######################################################################

if len(sys.argv) < 2:
    print("Usage: {} <recorded SSTV wave file> [start] [stop]".format(sys.argv[0]))
    sys.exit(1)
elif len(sys.argv) == 2:
    samples = block_wave_file(sys.argv[1])
elif len(sys.argv) == 3:
    samples = block_wave_file(sys.argv[1], float(sys.argv[2]))
elif len(sys.argv) == 4:
    samples = block_wave_file(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))

samples = block_differentiate(samples)
samples = block_rectify(samples)
samples = block_lowpass_filter_fir(samples, 150.0)
stream = stream_samples_to_stream(samples)
image = decode_sstv(stream)
block_plot(samples)

image.save("image.png", "PNG")
print("Image saved to image.png.")

plt.grid(False)
plt.imshow(image)
plt.show()

