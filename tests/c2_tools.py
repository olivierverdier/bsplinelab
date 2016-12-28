import numpy as np

def bis(f, t, h):
    return (f(t+2*h)
            + f(t)
            - 2*f(t+h))/(h*h)

def generate_diffs(f, t, emin=2, emax=6):
    for k in range(emin,emax):
        h = 10.**(-k)
        bbis_diff = bis(f,t,h) - bis(f,t,-h)
        yield k, bbis_diff

def gen_log10_errors(f, t):
    for k,d in generate_diffs(f, t):
        err = np.log10(np.max(np.abs(d)))
        yield k, err
