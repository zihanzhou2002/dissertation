import numpy as np

def solv_harm_oscil(p0, q0, omega, t):   
    return p0*np.cos(omega*t) + q0/omega * np.sin(omega*t), q0*np.cos(omega*t) - p0 * omega * np.sin(omega*t)

def gen_harm_dataset(omega, L, T, num):
    p0s = np.random.uniform(-L, L, num)
    q0s = np.random.uniform(-L, L, num)
    
    time = np.random.uniform(0, T, num).reshape(-1,1)
    
    y = np.zeros((num, 2))
    
    for i in range(num):
        p0 = p0s[i]
        q0 = q0s[i]
        pt, qt = solv_harm_oscil(p0, q0, omega, time[i])
        y[i][0] = pt
        y[i][1] = qt
    
    X_func = np.stack((p0s, q0s), axis = 1)
    X = (X_func.astype(np.float32), time.astype(np.float32))
    return X, y.astype(np.float32)
    
def gen_harm_dataset_fixed(p0, q0, omega, T, num):
    X_func = np.zeros((num, 2))
    time = np.linspace(0, T, num).reshape(-1,1)
    y = np.zeros((num, 2))
    
    for i in range(num):
        pt, qt = solv_harm_oscil(p0, q0, omega, time[i])
        y[i][0] = pt
        y[i][1] = qt
        
        X_func[i][0] = p0
        X_func[i][1] = q0
    
    X = (X_func.astype(np.float32), time.astype(np.float32))
    return X, y.astype(np.float32)

def gen_harm_dataset_mult_time(omega, L, T, num, num_t):
    p0s = np.random.uniform(-L, L, num)
    q0s = np.random.uniform(-L, L, num)
    
    time = np.random.uniform(0, T, num).reshape(-1,1)
    
    y = np.zeros((num, 2))
    
    for i in range(num):
        p0 = p0s[i]
        q0 = q0s[i]
        pt, qt = solv_harm_oscil(p0, q0, omega, time[i])
        y[i][0] = pt
        y[i][1] = qt
    
    X_func = np.stack((p0s, q0s), axis = 1)
    X = (X_func.astype(np.float32), time.astype(np.float32))
    return X, y.astype(np.float32)