from typing import List
import numpy as np
from numpy.typing import NDArray


def posterior_decode(observation_seq: List[int], hmm) -> List[str]:

    post = _posterior_probabilities(observation_seq, hmm)     
    idx = np.argmax(post, axis=1)
    return [hmm.states[i] for i in idx.tolist()]


def _posterior_probabilities(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    F = _build_forward_matrix(observation_seq, hmm)   
    B = _build_backward_matrix(observation_seq, hmm)  
    P = F * B                                         

    nan_rows = np.any(np.isnan(P), axis=1, keepdims=True) 


    row_sums = np.nansum(P, axis=1, keepdims=True)        

    ok = (~nan_rows) & (row_sums > 0)                     

    P = np.divide(P, row_sums, out=np.full_like(P, np.nan), where=ok)

    return P

def _build_forward_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:
 
    T = len(observation_seq)
    S = len(hmm.states)
    A: NDArray[np.float64] = np.asarray(hmm.transition_matrix, dtype=float)   
    B: NDArray[np.float64] = np.asarray(hmm.emission_matrix, dtype=float)     
    pi: NDArray[np.float64] = np.asarray(hmm.initial_state_probs, dtype=float) 

    F = np.zeros((T, S), dtype=float)


    o0 = observation_seq[0]
    F[0, :] = pi * B[o0, :]                        
    _normalize_row_(F, 0)


    for t in range(1, T):
        ot = observation_seq[t]
     
        temp = F[t-1, :].dot(A)                    
        F[t, :] = temp * B[ot, :]                 
        _normalize_row_(F, t)
    return F

def _build_backward_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:

    T = len(observation_seq)
    S = len(hmm.states)
    A: NDArray[np.float64] = np.asarray(hmm.transition_matrix, dtype=float)   
    B: NDArray[np.float64] = np.asarray(hmm.emission_matrix, dtype=float)    

    Bt = np.zeros((T, S), dtype=float)
    Bt[T-1, :] = 1.0
    _normalize_row_(Bt, T-1)

    for t in range(T-2, -1, -1):
        o_next = observation_seq[t+1]

        temp = B[o_next, :] * Bt[t+1, :]          

        Bt[t, :] = A.dot(temp)                   
        _normalize_row_(Bt, t)
    return Bt

def _normalize_row_(M: np.ndarray, row_idx: int) -> None:
    
    s = float(np.sum(M[row_idx, :]))
    if s > 0.0:
        M[row_idx, :] /= s
    else:
        M[row_idx, :] = np.nan

def _max_position(list_of_numbers: NDArray[np.float64]) -> int:
    max_value = -np.inf
    max_position = 0
    for i, value in enumerate(list_of_numbers):
        if max_value > 0:
            if value / max_value > 1 + 1e-5:
                max_value = value
                max_position = i
        else:
            if value > max_value:
                max_value = value
                max_position = i
    return max_position
