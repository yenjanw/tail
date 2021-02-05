cimport cython
import numpy as np
cimport numpy as cnp
ctypedef cnp.double_t DTYPE_t
from math import sqrt
import talib.stream as tas
import bottleneck as bn

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class MAMA:
    cdef:
        dict __dict__
        readonly double slowLimit, fastLimit
        # readonly cnp.ndarray s, d, p, sp, ph, q1, q2, i1, i2,re, im, mama, fema, data
        readonly list s, d, p, sp, ph, q1, q2, i1, i2,re, im, mama, fema, data
    def __cinit__(self, double fastLimit=0.15, double slowLimit=0.05):
        self.fastLimit, self.slowLimit = fastLimit, slowLimit
        self.s, self.d = [0] * 7, [0] * 7
        self.p, self.sp, self.ph = [0] * 3, [0] * 2, [0] * 2
        self.q1, self.q2 = [0] * 7, [0] * 2
        self.i1, self.i2 = [0] * 7, [0] * 2
        self.re, self.im = [0] * 7, [0] * 2
        self.mama_c, self.fema_c = [0] * 6, [0] * 6
        self.data = [0] * 4

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef add_one(self, double data):
        self.data[:3] = self.data[1:]
        self.data[3] = data
        if self.data[0] == 0: return None, None

        self.s[:6] = self.s[1:]
        self.s[6] = (4 * self.data[3] + 3 * self.data[2] + 2 * self.data[1] + self.data[0]) / 10
        self.d[:6] = self.d[1:]
        self.d[6] = (0.0962 * self.s[6] + 0.5769 * self.s[4] - 0.5769 * self.s[2] - 0.0962 * self.s[0]) * (
                    0.075 * self.p[1] + 0.54)
        # Compute InPhase and Quadrature components
        self.q1[:6] = self.q1[1:]
        self.q1[6] = (0.0962 * self.d[6] + 0.5769 * self.d[4] - 0.5769
                      * self.d[2] - 0.0962 * self.d[0]) * (0.075 * self.p[1] + 0.54)
        self.i1[:6] = self.i1[1:]
        self.i1[6] = self.d[3]
        # Advance the phase of I1 and Q1 by 90 degrees
        cdef double ji = (0.0962 * self.i1[6] + 0.5769 * self.i1[4] - 0.5769 * self.i1[2]
                          - 0.0962 * self.i1[0]) * (0.075 * self.p[1] + 0.54)
        cdef double jq = (0.0962 * self.q1[6] + 0.5769 * self.q1[4] - 0.5769 * self.q1[2]
                          - 0.0962 * self.q1[0]) * (0.075 * self.p[1] + 0.54)
        # Phasor addition for 3 bar averaging
        cdef double _i2 = self.i1[6] - jq
        cdef double _q2 = self.q1[6] + ji
        # Smooth the I and Q components before applying the discriminator
        self.i2[0] = self.i2[1]
        self.i2[1] = 0.2 * _i2 + 0.8 * self.i2[0]
        self.q2[0] = self.q2[1]
        self.q2[1] = 0.2 * _q2 + 0.8 * self.q2[0]
        # Homodyne Discriminator
        cdef double _re = self.i2[1] * self.i2[0] + self.q2[1] * self.q2[0]
        cdef double _im = self.i2[1] * self.q2[0] + self.q2[1] * self.i2[0]
        self.re[0] = self.re[1]
        self.re[1] = 0.2 * _re + 0.8 * self.re[0]
        self.im[0] = self.im[1]
        self.im[1] = 0.2 * _im + 0.8 * self.im[0]
        # set period value
        cdef double period = 0
        if _im != 0 and _re != 0: period = 360 / np.arctan(_im / _re)
        if period > 1.5 * self.p[1]: period = 1.5 * self.p[1]
        if period < 0.67 * self.p[1]: period = 0.67 * self.p[1]
        if period < 6: period = 6
        if period > 50: period = 50
        self.p[:2] = self.p[1:]
        self.p[2] = 0.2 * period + 0.8 * self.p[0]
        self.sp[0] = self.sp[1]
        self.sp[1] = 0.33 * self.p[0] + 0.67 * self.sp[0]
        self.ph[0] = self.ph[1]
        if self.i1[6] != 0: self.ph[1] = np.arctan(self.q1[6] / self.i1[6])
        # delta phase
        cdef double deltaPhase = self.ph[0] - self.ph[1]
        if deltaPhase < 1: deltaPhase = 1
        # alpha
        cdef double alpha = self.fastLimit / deltaPhase
        if alpha < self.slowLimit: alpha = self.slowLimit
        # add to output using EMA formula
        self.mama_c[:5] = self.mama_c[1:]
        self.mama_c[5] = alpha * self.data[3] + (1 - alpha) * self.mama_c[5]
        self.fema_c[:5] = self.fema_c[1:]
        self.fema_c[5] = 0.5 * alpha * self.mama_c[5] + (1 - 0.5 * alpha) * self.fema_c[5]
        return self.mama_c[5], self.fema_c[5]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.profile(False)
    cpdef reset(self):
        self.s, self.d = [0] * 7, [0] * 7
        self.p, self.sp, self.ph = [0] * 3, [0] * 2, [0] * 2
        self.q1, self.q2 = [0] * 7, [0] * 2
        self.i1, self.i2 = [0] * 7, [0] * 2
        self.re, self.im = [0] * 7, [0] * 2
        self.mama_c, self.fema_c = [0] * 6, [0] * 6
        self.data = [0] * 4

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class ATR:
    cdef:
        dict __dict__
        readonly list tr, high, low, close
        readonly int timeperiod, point
    def __cinit__(self, int timeperiod=14):
        self.period = timeperiod
        self.point = self.period - 1
        self.tr = [0] * self.period
        self.high, self.low, self.close = [0] * 2, [0] * 2, [0] * 2
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef double add_one(self, double high, double low, double close):
        self.high[0], self.high[1] = self.high[1], high
        self.low[0], self.low[1] = self.low[1], low
        self.close[0], self.close[1] = self.close[1], close
        self.tr[0:self.point] = self.tr[1:]
        self.tr[self.point] = max(self.high[1] - self.low[1], abs(self.high[1] - self.close[0]), abs(self.low[1] - self.close[0]))
        return sum(self.tr) / len(self.tr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.profile(False)
    cpdef reset(self):
        self.tr = [0] * self.period
        self.high, self.low, self.close = [0] * 2, [0] * 2, [0] * 2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class HMA:
    cdef:
        dict __dict__
        readonly int period, period2
        readonly object wma1, wma2, hma_wma
    def __cinit__(self, int timeperiod=12):
        self.period = timeperiod
        self.period2 = round(self.period/2)
        self.wma1 = WMA(self.period)
        self.wma2 = WMA(self.period2)
        self.hma_wma = WMA(round(sqrt(self.period)))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef double add_one(self, double data):
        return self.hma_wma.add_one(self.wma2.add_one(data) * 2 - self.wma1.add_one(data))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.profile(False)
    cpdef reset(self):
        self.wma1 = WMA(self.period)
        self.wma2 = WMA(self.period2)
        self.hma_wma = WMA(round(sqrt(self.period)))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class KAMA:
    cdef:
        dict __dict__
        readonly list data
        readonly double fast, slow, kama
        readonly int period, point
    def __cinit__(self, int timeperiod=10, double fast=2.0, double slow=30.0):
        self.period = timeperiod
        self.fast = fast
        self.slow = slow
        self.data = [0] * self.period
        self.kama = 0.0
        self.point = self.period - 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef double add_one(self, double data):
        self.data[:self.point] = self.data[1:]
        self.data[self.point] = data
        cdef:
            double fast_factor = 2.0 / (self.fast + 1.0)  # fast ema smoothing factor
            double slow_factor = 2.0 / (self.slow + 1.0)  # slow ema smoothing factor
            double direction = self.data[self.point] - self.data[0]
            double volatility = sum(np.abs(np.diff(self.data, n=1)))
            double er = abs(direction / volatility) if volatility != 0 else 0
            double SC = ((er * (fast_factor - slow_factor)) + slow_factor) ** 2  # scalable constant
        self.kama = self.kama + SC * (self.data[self.point] - self.kama)
        return self.kama

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.profile(False)
    cpdef reset(self):
        self.data = [0] * self.period
        self.kama = 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class ADX:
    cdef:
        dict __dict__
        # readonly cnp.ndarray s, d, p, sp, ph, q1, q2, i1, i2,re, im, high, low, close, atr
        readonly list s, d, p, sp, ph, q1, q2, i1, i2,re, im, high, low, close, atr
        readonly int period
        readonly object ATR
    def __cinit__(self, timeperiod=14):
        self.period = timeperiod
        self.pdm, self.mdm, self.spdm, self.smdm, self.dx, self.adx, self.pdi, self.mdi = \
            [0] * self.period, [0] * self.period, [0] * self.period, [0] * self.period, [0] * self.period, \
            [0] * 5, [0] * 5, [0] * 5
        self.high, self.low, self.close, self.atr = [0] * 2, [0] * 2, [0] * 2, [0] * 2
        self.ATR = ATR(timeperiod=self.period)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef add_one(self, double high, double low, double close):
        self.high[0], self.high[1] = self.high[1], high
        self.low[0], self.low[1] = self.low[1], low
        self.close[0], self.close[1] = self.close[1], close
        self.atr[0] = self.atr[1]
        self.atr[1] = self.ATR.add_one(high=self.high[1], low=self.low[1], close=self.close[1])
        cdef:
            double dm_p = self.high[1] - self.high[0]
            double dm_m = self.low[0] - self.low[1]
        self.pdm[:len(self.pdm) - 1] = self.pdm[1:]
        self.pdm[len(self.pdm) - 1] = dm_p if dm_p > dm_m and dm_p > 0 else 0

        self.mdm[:len(self.mdm) - 1] = self.mdm[1:]
        self.mdm[len(self.mdm) - 1] = dm_m if dm_m > dm_p and dm_m > 0 else 0

        self.spdm[:len(self.spdm) - 1] = self.spdm[1:]
        self.smdm[:len(self.smdm) - 1] = self.smdm[1:]
        if self.pdm[0] == 0:
            self.spdm[len(self.spdm) - 1] = sum(self.pdm) / len(self.pdm)
            self.smdm[len(self.smdm) - 1] = sum(self.mdm) / len(self.mdm)
        else:
            self.spdm[len(self.spdm) - 1] = (self.spdm[len(self.spdm) - 2] * (self.period - 1)
                                             + self.pdm[len(self.pdm) - 1]) / len(self.spdm)
            self.smdm[len(self.smdm) - 1] = (self.smdm[len(self.smdm) - 2] * (self.period - 1)
                                             + self.mdm[len(self.mdm) - 1]) / len(self.smdm)

        self.pdi[:len(self.pdi) - 1] = self.pdi[1:]
        self.pdi[len(self.pdi) - 1] = 100.0 * self.spdm[len(self.spdm) - 1] / float(self.atr[1])
        self.mdi[:len(self.mdi) - 1] = self.mdi[1:]
        self.mdi[len(self.mdi) - 1] = 100.0 * self.smdm[len(self.smdm) - 1] / float(self.atr[1])

        self.dx[:len(self.dx) - 1] = self.dx[1:]
        self.dx[len(self.dx) - 1] = 100.0 * float(abs(self.pdi[len(self.pdi) - 1]
                                                      - self.mdi[len(self.mdi) - 1])) \
                                    / (self.pdi[len(self.pdi) - 1] + self.mdi[len(self.mdi) - 1])
        self.adx[:len(self.adx) - 1] = self.adx[1:]
        if self.dx[0] == 0:
            self.adx[len(self.adx) - 1] = sum(self.dx) / float(self.period)
        else:
            self.adx[len(self.adx) - 1] = (self.adx[len(self.adx) - 1] * (self.period - 1)
                                           + self.dx[len(self.dx) - 1]) / float(self.period)
        return self.pdi[len(self.pdi)-1], self.mdi[len(self.mdi)-1], self.adx[4]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef reset(self):
        self.pdm, self.mdm, self.spdm, self.smdm, self.dx, self.adx, self.pdi, self.mdi = \
            [0] * self.period, [0] * self.period, [0] * self.period, [0] * self.period, [0] * self.period, \
            [0] * 5, [0] * 5, [0] * 5
        self.high, self.low, self.close, self.atr = [0] * 2, [0] * 2, [0] * 2, [0] * 2
        self.ATR = ATR(timeperiod=self.period)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class STOCH:
    cdef:
        dict __dict__
        readonly int period, point
        readonly list high, low, close
        readonly double k, d
    def __cinit__(self, timeperiod=9):
        self.period = timeperiod
        self.high, self.low, self.close = [0] * self.period, [0] * self.period, [0] * self.period
        self.point = self.period - 1
        self.k = 50.0
        self.d = 50.0
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef add_one(self, double high, double low, double close):
        self.high[:self.point], self.high[self.point] = self.high[1:], high
        self.low[:self.point], self.low[self.point] = self.low[1:], low
        self.close[:self.point], self.close[self.point] = self.close[1:], close
        self.k =  ((self.close[self.period-1] - min(self.low)) / (max(self.high) - min(self.low))) / 3 + 2 / 3 * self.k
        self.d = self.k / 3 + 2 / 3 * self.d
        return self.k, self.d

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef reset(self):
        self.high, self.low, self.close = [0] * self.period, [0] * self.period, [0] * self.period
        self.k = 50.0
        self.d = 50.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class RSI:
    """
    RSI
    input:[double] close
    output: real
    """
    cdef:
        dict __dict__
        readonly int period, point, count
        readonly double close, up_avg, down_avg, total
        # readonly cnp.ndarray data
        readonly list data
    def __cinit__(self, timeperiod=14):
        self.period = timeperiod
        self.point = self.period - 1
        self.data = [0] * self.period
        self.close = 0.0
        self.up_avg, self.down_avg, self.total = 0.0, 0.0, 0.0
        self.count = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef double add_one(self, double close):
        self.data[:self.point], self.data[self.point] = self.data[1:], close - self.close
        self.close = close
        cdef:
            double gain = 0.0
            double loss = 0.0
        self.count += 1
        if self.count <= self.period + 1: return 0
        elif self.count == self.period:
            self.up_avg = sum(self.data[self.data > 0]) / self.period
            self.down_avg = -1 * sum(self.data[self.data < 0]) / self.period
            self.total = self.up_avg + self.down_avg
            self.total = 1.0 if self.total == 0 else self.total
            return self.up_avg / self.total
        else:
            gain = self.data[self.point] if self.data[self.point] > 0 else 0.0
            loss = -1 * self.data[self.point] if self.data[self.point] < 0 else 0.0
            self.count = self.period + 2
            self.up_avg = (self.up_avg * self.point + gain) / self.period
            self.down_avg = (self.down_avg * self.point + loss) / self.period
            if self.down_avg == 0: return 1
            return 1.0 - (1.0 / (1.0 + self.up_avg / self.down_avg))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef reset(self):
        self.data = [0] * self.period
        self.close = 0.0
        self.up_avg, self.down_avg, self.total = 0.0, 0.0, 0.0
        self.count = 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class SAR:
    """
    SAR
    input:[double] high, low, close
    output: real
    """
    cdef:
        dict __dict__
        readonly double af_up, af_down, ep_up, ep_down, ep0, sr_value, init, acc, max
        readonly int position
        readonly list high, low, close
    def __cinit__(self, init=0.02, acc=0.02, max=0.2):
        self.high, self.low, self.close = [0] * 2, [0] * 2, [0] * 2
        self.init = init
        self.acc = acc
        self.max = max
        self.af_up, self.ep_up = 0.02, 0.0
        self.af_down, self.ep_down = 0.02, 0.0
        self.ep0 = 0.0
        self.sr_value = 0.0
        self.position = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef double add_one(self, double high, double low, double close):
        self.high[0],  self.high[1]  = self.high[1],  high
        self.low[0],   self.low[1]   = self.low[1],   low
        self.close[0], self.close[1] = self.close[1], close
        if self.close[0] == 0: return 0

        if self.position == 0:
            if self.close[1] > self.close[0]:  # 上昇
                self.af_up, self.ep_up = self.init, max(self.high)
                self.sr_value = min(self.low)
                if self.high[1] >= self.ep0: self.ep0 = self.high[1]
                self.position = 1
                return self.sr_value
            elif self.close[1] <= self.close[0]:  # 下降
                self.af_down, self.ep_down = self.init, max(self.low)
                self.sr_value = max(self.high)
                if self.low[1] <= self.ep0: self.ep0 = self.low[1]
                self.position = -1
                return self.sr_value

        elif self.position == 1:  # 上昇
            if self.low[1] >= self.sr_value:
                if self.high[1] > self.ep_up:
                    self.af_up += self.acc
                    self.ep_up = self.high[1]
                if self.af_up >= self.max: self.af_up = self.max
                self.sr_value = self.sr_value + self.af_up * (self.ep_up - self.sr_value)
                if self.high[1] >= self.ep0: self.ep0 = self.high[1]
                return self.sr_value

            elif self.low[1] < self.sr_value:
                self.sr_value = self.ep0
                # reset
                self.ep_down, self.af_down, self.af_up = self.ep0, self.init, self.init
                self.position = -1
                return self.sr_value

        elif self.position == -1:  # 下降
            if self.high[1] < self.sr_value:
                if self.low[1] < self.ep_down:
                    self.af_down += self.acc
                    self.ep_down = self.low[1]
                if self.af_down >= self.max: self.af_down = self.max
                self.sr_value = self.sr_value + self.af_down * (self.ep_down - self.sr_value)
                if self.low[1] <= self.ep0: self.ep0 = self.low[1]
                return self.sr_value
            elif self.high[1] >= self.sr_value:
                self.sr_value = self.ep0
                # reset
                self.ep_up, self.af_up, self.af_down = self.ep0, self.init, self.init
                self.position = 1
                return self.sr_value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef reset(self):
        self.high, self.low, self.close = [0] * 2, [0] * 2, [0] * 2
        self.af_up, self.ep_up = self.init, 0.0
        self.af_down, self.ep_down = self.init, 0.0
        self.ep0 = 0.0
        self.sr_value = 0.0
        self.position = 0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class EMA:
    cdef:
        dict __dict__
        readonly int period
        readonly double ema, mult
    def __cinit__(self, int timeperiod=14):
        self.period = timeperiod
        self.ema = 0.0
        self.mult = 2.0 / (self.period + 1)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef double add_one(self, double close):
        if self.ema == 0.0:
            self.ema = close
            return self.ema
        self.ema = self.mult * close + (1.0 - self.mult) * self.ema
        return self.ema
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef reset(self):
        self.ema = 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class MACD:
    cdef:
        dict __dict__
        readonly double fast, slow
        readonly object fast_ema, slow_ema, signal_ema
    def __init__(self):
        self.fast, self.slow = 0.0, 0.0

    def __cinit__(self, int fastperiod=12, int slowperiod=26, int signalperiod=9):
        self.fast_ema = EMA(timeperiod=fastperiod)
        self.slow_ema = EMA(timeperiod=slowperiod)
        self.signal_ema = EMA(timeperiod=signalperiod)
        self.fast, self.slow = 0.0, 0.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef add_one(self, double close):
        self.fast = self.fast_ema.add_one(close)
        self.slow = self.slow_ema.add_one(close)
        cdef double dif = self.fast - self.slow
        return dif, self.signal_ema.add_one(dif)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef reset(self):
        self.fast, self.slow = 0.0, 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class ICHMOKU:
    cdef:
        dict __dict__
        readonly int x, y, z, point
        readonly list high, low, close, tenkan, kijun, senkouaA, senkouB
    def __cinit__(self, int x=5, int y=10, int z=20):
        self.x = x
        self.y = y
        self.z = z
        self.high = [0] * (self.y + self.z + 1)
        self.low = [0] * (self.y + self.z + 1)
        self.close = [0] * (self.y + self.z + 1)
        self.point = self.y + self.z
        self.tenkan, self.kijun, self.senkouA, self.senkouB =  [0] * (self.y+1), [0] * (self.y+1), [0] * 5, [0] * 5

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef add_one(self, double high, double low, double close):
        self.high[:self.point], self.high[self.point] = self.high[1:], high
        self.low[:self.point], self.low[self.point] = self.low[1:], low
        self.close[:self.point], self.close[self.point] = self.close[1:], close

        self.tenkan[:len(self.tenkan)-1] = self.tenkan[1:]
        self.tenkan[len(self.tenkan)-1] = (max(self.high[len(self.high)-1-self.x:]) +
                                           min(self.low[len(self.low)-1-self.x:])) / 2.0
        self.kijun[:len(self.kijun)-1] = self.kijun[1:]
        self.kijun[len(self.kijun)-1] = (max(self.high[len(self.high)-1-self.y:]) +
                                         min(self.low[len(self.low)-1-self.y:])) / 2.0
        cdef:
            double senkouA = (self.tenkan[0] + self.kijun[0]) / 2.0
            double senkouB = (max(self.high[:len(self.high)-self.y-1]) + min(self.low[:len(self.low)-self.y-1])) / 2.0
        return self.tenkan[len(self.tenkan)-1], self.kijun[len(self.kijun)-1], senkouA, senkouB

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef reset(self):
        self.high = [0] * (self.y + self.z + 1)
        self.low = [0] * (self.y + self.z + 1)
        self.close = [0] * (self.y + self.z + 1)
        self.point = self.y + self.z
        self.tenkan, self.kijun, self.senkouA, self.senkouB = [0] * (self.y + 1), [0] * (self.y + 1), [0] * 5, [0] * 5

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class LLT:
    cdef:
        dict __dict__
        readonly double period
        # readonly cnp.ndarray data, llt
        readonly list data, llt
    def __cinit__(self, timeperiod=20):
        self.period = 2 / (timeperiod + 1)
        self.data = [0] * 3
        self.llt = [0] * 3

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef double add_one(self, double close):
        self.data[:2], self.data[2] = self.data[1:], close
        self.llt[:2] = self.llt[1:]
        self.llt[2] = (self.period - self.period ** 2 / 4) * self.data[2] \
                      + (self.period ** 2 / 2) * self.data[1] \
                      - (self.period - 3 * (self.period ** 2) / 4) * self.data[0] \
                      + 2 * (1 - self.period) * self.llt[1] \
                      - (1 - self.period) ** 2 * self.llt[0]
        return self.llt[2]
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef reset(self):
        self.data = [0] * 3
        self.llt = [0] * 3

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class WMA:
    cdef:
        dict __dict__
        readonly list data
        readonly double data_down_sum
        readonly int period, point
    def __cinit__(self, timeperiod=14):
        self.period = timeperiod
        self.data = [0] * self.period
        self.point = self.period - 1
        self.data_sum = self.period * (self.period + 1) / 2
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef double add_one(self, double close):
        self.data[:self.point], self.data[self.point] = self.data[1:], close
        cdef double total = 0.0
        for i in range(self.period): total += self.data[i] * (i + 1)
        return total / self.data_sum
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef reset(self):
        self.data = [0] * self.period

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class PATTEN:
    cdef:
        dict __dict__
        readonly cnp.ndarray open, high, low, close
    def __cinit__(self):
        self.open, self.high, self.low, self.close = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef list add_one(self, double open, double high, double low, double close):
        cdef list obs = [0] * 61
        self.open[:4], self.open[4] = self.high[1:], open
        self.high[:4], self.high[4] = self.high[1:], high
        self.low[:4], self.low[4] = self.low[1:], low
        self.close[:4], self.close[4] = self.close[1:], close
        obs[0] = tas.CDL2CROWS(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[1] = tas.CDL3BLACKCROWS(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[2] = tas.CDL3INSIDE(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[3] = tas.CDL3LINESTRIKE(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[4] = tas.CDL3OUTSIDE(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[5] = tas.CDL3STARSINSOUTH(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[6] = tas.CDL3WHITESOLDIERS(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[7] = tas.CDLABANDONEDBABY(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[8] = tas.CDLADVANCEBLOCK(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[9] = tas.CDLBELTHOLD(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[10] = tas.CDLBREAKAWAY(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[11] = tas.CDLCLOSINGMARUBOZU(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[12] = tas.CDLCONCEALBABYSWALL(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[13] = tas.CDLCOUNTERATTACK(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[14] = tas.CDLDARKCLOUDCOVER(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[15] = tas.CDLDOJI(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[16] = tas.CDLDOJISTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[17] = tas.CDLDRAGONFLYDOJI(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[18] = tas.CDLENGULFING(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[19] = tas.CDLEVENINGDOJISTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[20] = tas.CDLEVENINGSTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[21] = tas.CDLGAPSIDESIDEWHITE(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[22] = tas.CDLGRAVESTONEDOJI(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[23] = tas.CDLHAMMER(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[24] = tas.CDLHANGINGMAN(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[25] = tas.CDLHARAMI(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[26] = tas.CDLHARAMICROSS(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[27] = tas.CDLHIGHWAVE(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[28] = tas.CDLHIKKAKE(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[29] = tas.CDLHIKKAKEMOD(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[30] = tas.CDLHOMINGPIGEON(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[31] = tas.CDLIDENTICAL3CROWS(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[32] = tas.CDLINNECK(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[33] = tas.CDLINVERTEDHAMMER(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[34] = tas.CDLKICKING(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[35] = tas.CDLKICKINGBYLENGTH(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[36] = tas.CDLLADDERBOTTOM(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[37] = tas.CDLLONGLEGGEDDOJI(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[38] = tas.CDLLONGLINE(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[39] = tas.CDLMARUBOZU(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[40] = tas.CDLMATCHINGLOW(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[41] = tas.CDLMATHOLD(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[42] = tas.CDLMORNINGDOJISTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[43] = tas.CDLMORNINGSTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[44] = tas.CDLONNECK(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[45] = tas.CDLPIERCING(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[46] = tas.CDLRICKSHAWMAN(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[47] = tas.CDLRISEFALL3METHODS(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[48] = tas.CDLSEPARATINGLINES(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[49] = tas.CDLSHOOTINGSTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[50] = tas.CDLSHORTLINE(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[51] = tas.CDLSPINNINGTOP(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[52] = tas.CDLSTALLEDPATTERN(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[53] = tas.CDLSTICKSANDWICH(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[54] = tas.CDLTAKURI(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[55] = tas.CDLTASUKIGAP(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[56] = tas.CDLTHRUSTING(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[57] = tas.CDLTRISTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[58] = tas.CDLUNIQUE3RIVER(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[59] = tas.CDLUPSIDEGAP2CROWS(open=self.open, high=self.high, low=self.low, close=self.close)
        obs[60] = tas.CDLXSIDEGAP3METHODS(open=self.open, high=self.high, low=self.low, close=self.close)
        return obs
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef reset(self):
        self.open, self.high, self.low, self.close = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class BBI:
    cdef:
        dict __dict__
        readonly int period, ma1_t, ma2_t, ma3_t, ma4_t, point
        readonly list data
    def __cinit__(self, timeperiod=3):
        self.period = timeperiod
        self.ma1_t = self.period
        self.ma2_t = self.period * 2
        self.ma3_t = self.period * 3
        self.ma4_t = self.period * 4
        self.data = [0] * self.ma4_t
        self.point = self.ma4_t -1
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef add_one(self, double close):
        self.data[:self.point] = self.data[1:]
        self.data[self.point] = close
        cdef:
            double ma1 = bn.move_mean(self.data, self.ma1_t, self.ma1_t)[self.point]
            double ma2 = bn.move_mean(self.data, self.ma2_t, self.ma2_t)[self.point]
            double ma3 = bn.move_mean(self.data, self.ma3_t, self.ma3_t)[self.point]
            double ma4 = bn.move_mean(self.data, self.ma4_t, self.ma4_t)[self.point]
        return (ma1 + ma2 + ma3 + ma4) / 4
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef reset(self):
        self.data = [0] * self.ma4_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
@cython.profile(False)
cdef class RUD:
    cdef:
        dict __dict__
    def __cinit__(self, int timeperiod=3):
        self.period = timeperiod
        self.point = self.period - 1
        self.open = [0] * self.period
        self.high = [0] * self.period
        self.low = [0] * self.period

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef add_one(self, double open, double high, double low):
        self.open[:self.point], self.open[self.point] = self.open[1:], open
        self.high[:self.point], self.high[self.point] = self.high[1:], high
        self.low[:self.point], self.low[self.point] = self.low[1:], low
        if self.open[0] != 0:
            return ((max(self.high) - self.open[0]) - (self.open[0] - min(self.low))) / 100
        else: return None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(False)
    @cython.profile(False)
    cpdef reset(self):
        self.open = [0] * self.period
        self.high = [0] * self.period
        self.low = [0] * self.period
