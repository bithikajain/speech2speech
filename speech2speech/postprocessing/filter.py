def bandpass_pycbc(signal, sr, low_freq_cutoff, high_freq_cutoff, lowpass_order = 8):
   from pycbc.types import TimeSeries
   ts = TimeSeries(signal, delta_t=1./sr)

   import pycbc.filter
   ts = pycbc.filter.highpass(ts, low_freq_cutoff)

   import numpy
   return numpy.array(pycbc.filter.lowpass_fir(ts, high_freq_cutoff, lowpass_order))
