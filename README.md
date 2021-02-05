# tail (Technical Analysis Incremental Library )


## What is this?
When we use standard tool [Ta-Lib](https://github.com/mrjbq7/ta-lib) to anysis time series data in training AI model, I found it is too slow to train because Ta-Lib always compute data from data beginning. So I just try to write a simple code in cython to speed up the training.  

## How to install?
1. Install `cython`. You can find document [here](https://cython.readthedocs.io/en/latest/src/quickstart/install.html).  
1. Download `tail_C.pyx` and `setup.py` in your folder.  
1. Terminal in the folder typing `python setup.py build_ext --inplace`

## How to use?
It's easy to use.  
1. Import the function and set `timeperiod`.  
2. Use `add_one` input some data.
3. When traing end of data, you can `reset` the function data.  
Let me make some example.  

### Example KAMA:
```
import numpy as np
from tali_C import KAMA
KAMA = KAMA()
kama = []

data = np.random.randint(1000, 1000) # generate some data

for i in range(len(data)):
    tmp = KAMA.add_one(data[i]) # use function 'add_one' to input a data
    kama.append(tmp)
KAMA.reset() # reset KAMA data
print(kama)
```

### Example ATR:
```
import numpy as np
import pandas as pd
from tali_C import ATR
ATR = ATR(timeperiod=14) # set ATR timeperiod
atr = []

df = pd.read_csv('twii.csv') # twii.csv is stock index with OHLC, just replace your data.

for i in range(len(df)):
    tmp = ATR.add_one(high = df.iloc[i]['high'], low = df.iloc[i]['low'], close = df.iloc[i]['close']) # ATR use High, Low, Close 
    atr.append(tmp)
ATR.reset() # reset ATR data
print(atr)
```


## What Indicators in this project.
1. MAMA
1. HMA
1. KAMA
1. ADX
1. STOCH(KD)
1. RSI
1. SAR
1. EMA
1. MACD
1. ICHMOKU
1. LLT
1. WMA
