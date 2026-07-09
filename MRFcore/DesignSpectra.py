import matplotlib.pyplot as plt
import numpy as np

def DesignSpectra():
    # Example parameters (these values need to be updated with real data)
    SDS = 1.0   # Short-period design spectral acceleration
    SD1 = 0.6   # 1-second period design spectral acceleration

    # Calculate key periods
    T0 = 0.2 * SD1 / SDS
    TS = SD1 / SDS

    # Define period values for the plot
    T = np.linspace(0, 4, 400)  # from 0 to 4 seconds
    Sa = np.piecewise(T, 
                      [ T <= T0, (T0 < T) & (T <= TS), T > TS], 
                      [lambda T: SDS * (0.4+0.6* T / T0), 
                       lambda T: SDS, 
                       lambda T: SD1 / T])
    return T , Sa

T1 , Sa1 = DesignSpectra()
T2 , Sa2 = T1 , 1.5 * Sa1