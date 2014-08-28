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


