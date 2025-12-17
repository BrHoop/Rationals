import numpy as np


class fixed_point:
    def __init__(self, FRAC_BITS):
        self.FRAC_BITS = FRAC_BITS
        self.SCALE = 1 << FRAC_BITS
        self.HALF = 1 << (FRAC_BITS - 1)
        self.FIXED_ONE = self.SCALE


    def _as_int_array(self,value):
        return np.asarray(value, dtype=np.int64)


    def _to_result(self,value):
        arr = self._as_int_array(value)
        if arr.shape == ():
            return int(arr)
        return arr.astype(np.int64)


    def to_fixed_scalar(self, value):
        return int(round(float(value) * self.SCALE))


    def to_fixed_array(self, values):
        return np.rint(values * self.SCALE).astype(np.int64)


    def from_fixed_scalar(self, value):
        return float(value) / self.SCALE


    def from_fixed_array(self,values):
        return values.astype(np.float64) / self.SCALE


    def fixed_mul(self,a, b):
        a_arr = self._as_int_array(a)
        b_arr = self._as_int_array(b)
        result = (a_arr * b_arr + self.HALF) >> self.FRAC_BITS #The addition of the Half Scale unit makes it so that it rounds to the nearest reather than truncating
        if result.shape == ():
            return int(result)
        return result.astype(np.int64)


    def fixed_div(self,numerator, denominator):
        num = self._as_int_array(numerator)
        den = self._as_int_array(denominator)
        result = ((num << self.FRAC_BITS) + den // 2) // den
        if result.shape == ():
            return int(result)
        return result.astype(np.int64)


    def fixed_div_int(self,value, divisor):
        arr = self._as_int_array(value)
        result = (arr + divisor // 2) // divisor
        if result.shape == ():
            return int(result)
        return result.astype(np.int64)