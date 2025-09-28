import numpy as np

def data_generator(Num_samples, Phi, S, Sigma, Rice_K):
  
    if Rice_K <=  8:  # QPSK/4-QAM
        temp = np.random.randint(2, size=(2, Num_samples)) * 2 - 1
        Symbol_Tx = (temp[0, :] + 1j * temp[1, :]) / np.sqrt(2)
        
    elif 8 < Rice_K <= 14:  # 8-PSK
        phases = np.arange(0, 8) * np.pi / 4
        idx = np.random.randint(8, size=Num_samples)
        Symbol_Tx = np.exp(1j * phases[idx])
        
    elif Rice_K > 14:  # 16-QAM，使用固定的电平值数组[-3, -1, 1, 3]来生成16-QAM的星座点
        values = np.array([-3, -1, 1, 3])
        real_idx = np.random.randint(4, size=Num_samples)
        imag_idx = np.random.randint(4, size=Num_samples)
        Symbol_Tx = (values[real_idx] + 1j * values[imag_idx]) / np.sqrt(10)
        
    else:
        raise ValueError('Unsupported modulation!')
    
    # Generate noise
    noise = np.random.randn(Num_samples) * Sigma + 1j * np.random.randn(Num_samples) * Sigma
    
    # Received signal (Symbol_rx) and maximum likelihood estimation (rx_mle)
    Symbol_rx = Symbol_Tx * S * np.exp(1j * Phi) + noise
    rx_mle = S * np.exp(1j * Phi) + noise
    
    # Convert to real and imaginary parts
    dataMod = np.column_stack((np.real(Symbol_rx), np.imag(Symbol_rx)))
    DataUnmod = np.column_stack((np.real(rx_mle), np.imag(rx_mle)))
    
    return dataMod, DataUnmod
