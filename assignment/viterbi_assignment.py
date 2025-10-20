"""
Viterbi Decoding for Hidden Markov Models

This module implements the Viterbi algorithm for finding the most likely
state sequence given an observation sequence and an HMM.
"""

from typing import List
import numpy as np
from numpy.typing import NDArray


def viterbi_decode(observation_seq: List[int], hmm) -> List[str]:
    """
    Decode an observation sequence using the Viterbi algorithm.

    INPUT:
    - observation_seq: List of observations (integers indexing alphabet)
    - hmm: HMM object with states, initial_state_probs, transition_matrix,
      and emission_matrix

    OUTPUT:
    - state_seq: List containing the most likely state sequence (state names)

    IMPLEMENTATION NOTES:
    1) The Viterbi table is implemented as a matrix that is transposed
       relative to the way it is shown in class. Rows correspond to
       observations and columns to states.
    2) After computing the Viterbi probabilities for each observation,
       they are normalized by dividing by their sum. This maintains
       proportionality while avoiding numerical underflow.
    """
    numeric_state_seq = _traceback(
        _build_matrix(observation_seq, hmm),
        hmm
    )
    return [hmm.states[i] for i in numeric_state_seq]

def _build_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Build the Viterbi probability matrix.

    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """
    number_of_observations = len(observation_seq)
    number_of_states = hmm.num_states

   

    # YOUR CODE HERE
    number_of_observations = len(observation_seq)
    number_of_states = hmm.num_states

    # Initialize Viterbi matrix
    viterbi_matrix = np.zeros((number_of_observations, number_of_states))

    # YOUR CODE HERE
    T: int = len(observation_seq)                
    N: int = len(hmm.initial_state_probs)       

    # Allocate DP matrix
    V: NDArray[np.float64] = np.zeros((T, N), dtype=np.float64)

    # Model parameters (renamed for clarity; signature unchanged)
    initial: NDArray[np.float64]    = np.array(hmm.initial_state_probs, dtype=float)
    transition: NDArray[np.float64] = np.array(hmm.transition_matrix,    dtype=float)
    emission: NDArray[np.float64]   = np.array(hmm.emission_matrix,      dtype=float)

    # --- Initialization (t = 0) ---
    first_obs: int = observation_seq[0]
    V[0] = initial * emission[first_obs]

    row_sum = np.sum(V[0])
    if row_sum > 0:
        V[0] /= row_sum
    else:
        # If everything is zero, mark as NaN (matches original behavior)
        V[0] = np.full(N, np.nan)

    # --- Recursion (t = 1..T-1) ---
    for t in range(1, T):
        obs_t: int = observation_seq[t]
        row = np.zeros(N, dtype=np.float64)

        for s in range(N):
            # best previous path prob to state s times emission
            prev_scores = V[t - 1] * transition[:, s]
            row[s] = np.max(prev_scores) * emission[obs_t][s]

        row_sum = np.sum(row)
        if row_sum > 0:
            row /= row_sum
        else:
            row = np.ones(N, dtype=np.float64) / N

        V[t] = row


    if np.any(np.isnan(V)):
        V[:] = np.nan

    return V


def _traceback(viterbi_matrix: NDArray[np.float64], hmm) -> List[int]:
    """
    Trace back through the Viterbi matrix to find the most likely path.

    Returns a list of state indices (integers) corresponding to the
    most likely state sequence.
    """
    T = len(viterbi_matrix)  # number of observations/time steps
    state_sequence = np.zeros(T, dtype=int)

    transition = np.array(hmm.transition_matrix, dtype=float)
    state_sequence[-1] = _max_position(viterbi_matrix[-1])

    for t in range(T - 2, -1, -1):
        next_state = state_sequence[t + 1]
        scores = viterbi_matrix[t] * transition[:, next_state]
        state_sequence[t] = _max_position(scores)

    return state_sequence.tolist()

# Use this function to find the index within an array of the maximum value.
# Do not use any built-in functions for this.
# This implementation chooses the lowest index in case of ties.
# Two values are considered tied if they are within a factor of 1E-5.
def _max_position(list_of_numbers: NDArray[np.float64]) -> int:
    """
    Find the index of the maximum value in a list.

    Returns the first index if there are ties or extremly close values.
    """
    max_value = 1E-10
    max_position = 0

    for i, value in enumerate(list_of_numbers):
        # This handles extremely close values that arise from numerical instability
        if value / max_value > 1 + 1E-5:
            max_value = value
            max_position = i

    return max_position
