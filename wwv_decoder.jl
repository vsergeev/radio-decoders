using WAV
using DSP
using PyPlot

println("Libraries loaded...")

################################################################################

SAMPLE_RATE = None

################################################################################

function plot_filter(filter, hz=None)
    if hz == None
        freqs = 0:(SAMPLE_RATE/2)-1
    else
        freqs = hz
    end

    resp = 20*log10(abs(freqz(filter, freqs, SAMPLE_RATE)))
    plot(freqs, resp)
    readline()
end

################################################################################

function stream_wave_file(path, start=None, stop=None)
    global SAMPLE_RATE

    samples, SAMPLE_RATE = WAV.wavread(path)

    if start != None && stop == None
        samples = samples[int(start*SAMPLE_RATE)+1:end]
    elseif start != None && stop != None
        samples = samples[int(start*SAMPLE_RATE)+1:int(stop*SAMPLE_RATE)]
    end

    return samples
end

function stream_rectify(stream)
    Task(function ()
        for s in stream
            produce(abs(s))
        end
    end)
end

function stream_bandpass_filter_iir(stream, fLow, fHigh; realtime=false)
    bpf = convert(TFFilter, digitalfilter(Bandpass(fLow, fHigh; fs=SAMPLE_RATE), Butterworth(3)))
    #plot_filter(bpf, 0:1000)
    b, a = coefb(bpf), coefa(bpf)

    if !realtime
        return filt(b, a, collect(Float64, stream))
    else
        Task(function ()
            x = zeros(length(b))
            y = zeros(length(a)-1)

            for sample in stream
                # y[n] = 1/a_0 * ( (b_0*x[n] + b_1*x[n-1] + b_2*x[n-2] + ... b_N*x[n-N])
                #                   - (a_0*y[n-1] + a_1*y[n-2] + a_2*y[n-3] + ...) )
                x = cat(1, [sample], x[1:end-1])
                y_n = (dot(x, b) - dot(y, a[2:end])) / a[1]
                y = cat(1, [y_n], y[1:end-1])
                produce(y_n)
            end
        end)
    end
end

function stream_lowpass_filter_iir(stream, fC; realtime=false)
    bpf = convert(TFFilter, digitalfilter(Lowpass(fC; fs=SAMPLE_RATE), Butterworth(4)))
    #plot_filter(bpf, 0:1000)
    b, a = coefb(bpf), coefa(bpf)

    if !realtime
        return filt(b, a, collect(Float64, stream))
    else
        Task(function ()
            x = zeros(length(b))
            y = zeros(length(a)-1)

            for sample in stream
                # y[n] = 1/a_0 * ( (b_0*x[n] + b_1*x[n-1] + b_2*x[n-2] + ... b_N*x[n-N])
                #                   - (a_0*y[n-1] + a_1*y[n-2] + a_2*y[n-3] + ...) )
                x = cat(1, [sample], x[1:end-1])
                y_n = (dot(x, b) - dot(y, a[2:end])) / a[1]
                y = cat(1, [y_n], y[1:end-1])
                produce(y_n)
            end
        end)
    end
end

function stream_static_threshold(stream, threshold)
    Task(function ()
        for sample in stream
            if sample > threshold
                produce(1)
            else
                produce(0)
            end
        end
    end)
end

function stream_threshold(stream)
    state = []
    i = 0
    Task(function ()
        while length(state) < int(1.0*SAMPLE_RATE)
            state = cat(1, state, [float64(consume(stream))])
        end
        for sample in stream
            if i == 0 || i % 8000 == 0
                smin, smax = quantile(state, [0.15, 0.85])
                threshold = ((smax-smin)/2.0) + smin
            end

            if state[1] > 0.03 && state[1] > threshold
                produce(1)
            else
                produce(0)
            end

            state = cat(1, state[2:end], [sample])
            i += 1
        end
    end)
end

function stream_pulse_widths(stream)
    Task(function ()
        sample_number = 0
        state, width = (0, 0)

        for sample in stream
            sample_number += 1
            # 0 0
            if state == 0 && sample == 0
                state, width = (0, 0)
            # 0 1
            elseif state == 0 && sample == 1
                offset = sample_number
                state, width = (1, 1)
            # 1 1
            elseif state == 1 && sample == 1
                state, width = (1, width+1)
            # 1 0
            elseif state == 1 && sample == 0
                produce((offset/SAMPLE_RATE, width/SAMPLE_RATE))
                state, width = (0, 0)
            end
        end
    end)
end

function stream_filter_pulse_widths(stream)
    Task(function ()
        state = []

        for (offset, width) in stream
            if length(state) == 0
                if width > 125e-3
                    state = [(offset, width)]
                end
            else
                # Emit the collected pulse if this one is 900ms past the beginning
                # of the collection
                if abs(offset - state[1][1]) > 900e-3
                    produce((state[1][1], sum(map((t) -> t[2], state))))
                    state = [(offset, width)]
                # Add this pulse to our collection if it's less than 800ms from the
                # end of the last pulse in our collection
                elseif abs(offset - state[end][1]) < 800e-3
                    state = cat(1, state, [(offset, width)])
                end
            end
        end

        # Flush the last pulse collection when input stream is terminated
        if length(state) > 0
            produce((state[1][1], sum(map((t) -> t[2], state))))
        end
    end)
end

function stream_pulse_widths_to_symbols(stream)
    Task(function ()
        # approximately equal means actual vs. +/-25% of expected
        approx_equal = (actual, expected) -> abs(actual - expected) < 0.25*expected

        for (offset, width) in stream
            # 800ms for a marker
            if approx_equal(width, 800e-3)
                produce((offset, "M"))
            # 500ms for a 1 bit
            elseif approx_equal(width, 500e-3)
                produce((offset, 1))
            # 200ms for a 0 bit
            elseif approx_equal(width, 200e-3)
                produce((offset, 0))
            # Invalid bit
            else
                produce((offset, "I"))
            end
        end
    end)
end

function stream_symbols_to_frame(stream)
    compare = (expected, actual) -> (expected == "B" && (actual == 0 || actual == 1)) || actual == expected
    template = [     0 , "B", "B", "B", "B", "B", "B",  0 , "M",
                    "B", "B", "B", "B",  0 , "B", "B", "B",  0 , "M",
                    "B", "B", "B", "B",  0 , "B", "B",  0 ,  0 , "M",
                    "B", "B", "B", "B",  0 , "B", "B", "B", "B", "M",
                    "B", "B",  0 ,  0 ,  0 ,  0 ,  0 , "B",  0 , "M", # FIXME
                    "B", "B", "B", "B", "B", "B", "B", "B", "B", "M"    ]
    state = repmat([None], 59)

    Task(function ()
        for (offset, symbol) in stream
            println((offset, symbol))
            state = cat(1, state[2:end], [(offset, symbol)])

            if !(None in state)
                # Check that the symbols are each 1.0s second +/-100ms apart
                offsets = collect(Float64, map((t) -> t[1], state))
                if !all(abs(diff(offsets) - 1.0) .< 100e-3)
                    continue
                end

                # Check for no invalid symbols
                symbols = map((t) -> t[2], state)
                if "I" in symbols
                    continue
                end

                # Check that the symbols match the template
                matches = map((i) -> compare(template[i], state[i][2]), 1:length(template))
                if !all(matches)
                    continue
                end

                # Emit the symbols of the valid frame
                println("Valid frame found!")
                produce(symbols)

                # Reset the state
                state = repmat([None], 59)
            end
        end
    end)
end

type WWVRecord
    DST1::Bool
    LSW::Bool
    Year::Int
    Minutes::Int
    Hours::Int
    DayOfYear::Int
    DUT1::Bool
    DST2::Bool
    UT1Corr::Float64
end

function stream_frame_to_wwv_record(stream)

    Task(function ()
        for frame in stream
            dst1 = bool(frame[2])
            lsw = bool(frame[3])
            year = 1*frame[4] + 2*frame[5] + 4*frame[6] + 8*frame[7] + 10*frame[51] + 20*frame[52] + 40*frame[53] + 80*frame[54]
            minutes = 1*frame[10] + 2*frame[11] + 4*frame[12] + 8*frame[13] + 10*frame[15] + 20*frame[16] + 40*frame[17]
            hours = 1*frame[20] + 2*frame[21] + 4*frame[22] + 8*frame[23] + 10*frame[25] + 20*frame[26]
            day_of_year = 1*frame[30] + 2*frame[31] + 4*frame[32] + 8*frame[33] + 10*frame[35] + 20*frame[36] + 40*frame[37] + 80*frame[38] + 100*frame[40] + 200*frame[41]
            dut1 = bool(frame[50])
            dst2 = bool(frame[55])
            ut1_corr = 0.1*frame[56] + 0.2*frame[57] + 0.3*frame[58]

            produce(WWVRecord(dst1, lsw, year, minutes, hours, day_of_year, dut1, dst2, ut1_corr))
        end
    end)
end

function stream_print_wwv_record(stream)
    for record in stream
        println(record)
    end
end

function stream_plot(stream, n=None)
    if n == None
        data = collect(Float64, stream)
    else
        # FIXME
    end

    plot(0:length(data)-1, data)
    readline()
end

################################################################################

if length(ARGS) == 0
    @printf "Usage: <WWV recording wave file> [start] [stop]\n"
    exit(1)
end

println("Loading WAV file...")
if length(ARGS) == 1
    s0 = stream_wave_file(ARGS[1])
elseif length(ARGS) == 2
    s0 = stream_wave_file(ARGS[1], float(ARGS[2]))
elseif length(ARGS) >= 3
    s0 = stream_wave_file(ARGS[1], float(ARGS[2]), float(ARGS[3]))
end

println("Starting processing...")
s1 = stream_bandpass_filter_iir(s0, 95.0, 105.0; realtime=true)
s2 = stream_rectify(s1)
s3 = stream_lowpass_filter_iir(s2, 5.0; realtime=true)
s4 = stream_static_threshold(s3, 0.03)
s5 = stream_pulse_widths(s4)
s6 = stream_filter_pulse_widths(s5)
s7 = stream_pulse_widths_to_symbols(s6)
s8 = stream_symbols_to_frame(s7)
s9 = stream_frame_to_wwv_record(s8)
stream_print_wwv_record(s9)

