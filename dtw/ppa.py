import numpy as np 

def agregate_aprox(ts_i,frame_size=2):
    ts_size=ts_i.shape[0]
    n_frames=ts_size / frame_size
    frames=[ ts_i[(j*frame_size):(j+1)*frame_size] 
                    for j in range(n_frames)]
    new_ts=[np.mean(frame_j) for frame_j in frames]
    return np.array(new_ts)