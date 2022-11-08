from joblib import Parallel, delayed
import time


def hi(x):
    time.sleep(x)
    print(x)
    # for x in range(10000):
    #     h = 4 + x


Parallel(n_jobs=-4, verbose=10)(delayed(hi)(i) for i in range(10))
