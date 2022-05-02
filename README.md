# Leetcode

```
import numpy as np
class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        n, r = divmod(abs(numerator), abs(denominator))
        sign = '-' if np.sign(numerator) * np.sign(denominator) < 0 else ''
        result = [sign + str(n), '.']
        stack = []
        while r not in stack:
            stack.append(r)
            n, r = divmod(r*10, abs(denominator))
            result.append(str(n))

        idx = stack.index(r)
        result.insert(idx+2, '(')
        result.append(')')
        return ''.join(result).replace('(0)', '').rstrip('.')
```
