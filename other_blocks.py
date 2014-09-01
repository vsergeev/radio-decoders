import numpy
import scipy.signal

def stream_correlate(stream):
    t = numpy.array(range(int(200e-3*SAMPLE_RATE)))
    x = numpy.cos(((2*numpy.pi)/SAMPLE_RATE)*100*t)
    return numpy.correlate(stream, x)

def stream_lowpass_filter_fir_slow(stream, fC):
    b = scipy.signal.firwin(4096 + 1, (2*fC)/SAMPLE_RATE)
    #plot_filter(b, [1], SAMPLE_RATE, range(1000))

    state = numpy.array([0.0]*len(b))

    for sample in stream:
        # y[n] = b_0*x[n] + b_1*x[n-1] + b_2*x[n-2] + ... b_N*x[n-N]
        state = numpy.append(sample, state[:-1])
        yield numpy.inner(state, b)

def stream_bandpass_filter_fir_slow(stream, fLow, fHigh):
    b = scipy.signal.firwin(8192 + 1, [(2*fLow)/(SAMPLE_RATE), (2*fHigh)/(SAMPLE_RATE)], pass_zero=False)
    #plot_filter(b, [1], SAMPLE_RATE, range(1000))

    state = numpy.array([0.0]*len(b))

    for sample in stream:
        # y[n] = b_0*x[n] + b_1*x[n-1] + b_2*x[n-2] + ... b_N*x[n-N]
        state = numpy.append(sample, state[:-1])
        yield numpy.inner(state, b)

def stream_lowpass_filter_iir_slow(stream, fC):
    b, a = scipy.signal.butter(4, (2*fC)/SAMPLE_RATE)
    #plot_filter(b, a, SAMPLE_RATE, range(1000))

    x = numpy.array([0.0]*len(b))
    y = numpy.array([0.0]*(len(a)-1))

    for sample in stream:
        # y[n] = 1/a_0 * ( (b_0*x[n] + b_1*x[n-1] + b_2*x[n-2] + ... b_N*x[n-N])
        #                   - (a_0*y[n-1] + a_1*y[n-2] + a_2*y[n-3] + ...) )
        x = numpy.append(sample, x[:-1])
        y_n = (numpy.inner(x, b) - numpy.inner(y, a[1:]))/a[0]
        y = numpy.append(y_n, y[:-1])
        yield y_n

def stream_bandpass_filter_iir_slow(stream, fLow, fHigh, margin=10.0):
    b,a = scipy.signal.butter(3, [(2*fLow)/(SAMPLE_RATE), (2*fHigh)/(SAMPLE_RATE)], btype='bandpass')
    #plot_filter(b, a, SAMPLE_RATE, range(1000))

    x = numpy.array([0.0]*len(b))
    y = numpy.array([0.0]*(len(a)-1))

    for sample in stream:
        # y[n] = 1/a_0 * ( (b_0*x[n] + b_1*x[n-1] + b_2*x[n-2] + ... b_N*x[n-N])
        #                   - (a_0*y[n-1] + a_1*y[n-2] + a_2*y[n-3] + ...) )
        x = numpy.append(sample, x[:-1])
        y_n = (numpy.inner(x, b) - numpy.inner(y, a[1:]))/a[0]
        y = numpy.append(y_n, y[:-1])
        yield y_n


#    spectrogram = [numpy.copy(dft)]
#    W = numpy.exp((2*numpy.pi*1j*numpy.arange(0, N/2+1))/N)
#
#    for n in range(N,len(samples)):
#        # Marginally stable sliding DFT
#        # X_k(n) = [X_k(n-1) - x(n-N) + x(n)]*e^(2*pi*k/N)
#        dft = (dft - samples[n-N] + samples[n])*W
#        spectrogram.append(dft)
#
#    spectrogram = numpy.array(spectrogram)
#    spectrogram = numpy.abs(spectrogram.T)
#
#    plt.imshow(spectrogram, origin='lower', aspect='auto', interpolation='nearest')
#    plt.show()


    """
    wf = numpy.hanning(N)
    Wl = numpy.exp((-2*numpy.pi*1j*rtty_low_index*numpy.arange(N))/N)
    Wh = numpy.exp((-2*numpy.pi*1j*rtty_high_index*numpy.arange(N))/N)

    spectrogram = []
    for n in range(len(samples) - N - (len(samples) % N)):
        sample_window = numpy.array(samples[n:n+N])*wf
        dft_low = numpy.sum(sample_window * Wl)
        dft_high = numpy.sum(sample_window * Wh)
        spectrogram.append((dft_low, dft_high))
    """

    """
    dft = numpy.fft.rfft(samples, N)
    dft = numpy.array([dft[rtty_low_index], dft[rtty_high_index]])
    spectrogram = [numpy.copy(dft)]
    W = numpy.exp((2*numpy.pi*1j*numpy.array([rtty_low_index, rtty_high_index]))/N)

    for n in range(N,len(samples)):
        # Marginally stable sliding DFT
        # X_k(n) = [X_k(n-1) - x(n-N) + x(n)]*e^(2*pi*k/N)
        dft = (dft - samples[n-N] + samples[n])*W
        spectrogram.append(dft)
    """

