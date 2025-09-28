import numpy as np

def data_generator_from_k(Num_samples, Phi, Rice_K, modulation_type):
    S = 0.8
    Sigma = np.sqrt(S**2 / (2 * Rice_K))
    # Randomly select clusters value (4 or 16)
    if modulation_type == 'QPSK':
        clusters = 4
    
    elif modulation_type == '8PSK':
        clusters = 8
        
    elif modulation_type == '16QAM':
        clusters = 16
        
    else:
        raise ValueError('Unsupported modulation!')

    
    # Initialize temp based on clusters
    if clusters == 2:  # BPSK
        temp = np.random.randint(2, size=Num_samples) * 2 - 1
        Symbol_Tx = temp
        
    elif clusters == 4:  # QPSK/4-QAM
        temp = np.random.randint(2, size=(2, Num_samples)) * 2 - 1
        Symbol_Tx = (temp[0, :] + 1j * temp[1, :]) / np.sqrt(2)
        
    elif clusters == 8:  # 8-PSK
        phases = np.arange(0, 8) * np.pi / 4
        idx = np.random.randint(8, size=Num_samples)
        Symbol_Tx = np.exp(1j * phases[idx])
        
    elif clusters == 16:  # 16-QAM，使用固定的电平值数组[-3, -1, 1, 3]来生成16-QAM的星座点
        values = np.array([-3, -1, 1, 3])
        real_idx = np.random.randint(4, size=Num_samples)
        imag_idx = np.random.randint(4, size=Num_samples)
        Symbol_Tx = (values[real_idx] + 1j * values[imag_idx]) / np.sqrt(10)
        
    elif clusters == 64:  # 64-QAM
        values = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        real_idx = np.random.randint(8, size=Num_samples)
        imag_idx = np.random.randint(8, size=Num_samples)
        Symbol_Tx = (values[real_idx] + 1j * values[imag_idx]) / np.sqrt(42)  # Normalize energy
        
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
