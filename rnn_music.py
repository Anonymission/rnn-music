from mygrad import Tensor
from mygrad.nnet.activations import sigmoid, tanh
from mygrad.nnet.layers import dense
from mygrad.math import log
import numpy as np
import mido


def train_model_arr(data, k=5, hd=100, wub=None, hin1=None, hin2=None) :
    """
    OLD MODEL - TRAINS ON ARRAYS
    Trains model of RNN

    Parameters
    -----------
    data: ndarray
        array of data (what form?)
    k: int; default = 48 (.5 beats)
        number of backprops to undergo
    hd: int; default = 100
        size of hidden dimension
    wub: tuple of Tensors; optional
        weights from previous training
    hin: Tensor; optional
        output from previous training
    """
    md = len(data[0])
    wz1 = Tensor(he_normal((hd, hd))) if wub is None else wub[0, 0]
    uz1 = Tensor(he_normal((hd, md))) if wub is None else wub[0, 1]
    bz1 = Tensor(np.zeros((hd, 1), dtype=wz1.dtype)) if wub is None else wub[0, 2]
    wr1 = Tensor(he_normal((hd, hd))) if wub is None else wub[0, 3]
    ur1 = Tensor(he_normal((hd, md))) if wub is None else wub[0, 4]
    br1 = Tensor(np.zeros((hd, 1), dtype=wr1.dtype)) if wub is None else wub[0, 5]
    wh1 = Tensor(he_normal((hd, hd))) if wub is None else wub[0, 6]
    uh1 = Tensor(he_normal((hd, md))) if wub is None else wub[0, 7]
    bh1 = Tensor(np.zeros((hd, 1), dtype=wh1.dtype)) if wub is None else wub[0, 8]

    wz2 = Tensor(he_normal((hd, hd))) if wub is None else wub[1, 0]
    uz2 = Tensor(he_normal((hd, hd))) if wub is None else wub[1, 1]
    bz2 = Tensor(np.zeros((hd, 1), dtype=wz2.dtype)) if wub is None else wub[1, 2]
    wr2 = Tensor(he_normal((hd, hd))) if wub is None else wub[1, 3]
    ur2 = Tensor(he_normal((hd, hd))) if wub is None else wub[1, 4]
    br2 = Tensor(np.zeros((hd, 1), dtype=wr2.dtype)) if wub is None else wub[1, 5]
    wh2 = Tensor(he_normal((hd, hd))) if wub is None else wub[1, 6]
    uh2 = Tensor(he_normal((hd, hd))) if wub is None else wub[1, 7]
    bh2 = Tensor(np.zeros((hd, 1), dtype=wh2.dtype)) if wub is None else wub[1, 8]

    v = Tensor(he_normal((md, hd))) if wub is None else wub[2]
    c = Tensor(np.zeros((md, 1), dtype=wh2.dtype)) if wub is None else wub[3]

    wub = ((wz1, uz1, bz1, wr1, ur1, br1, wh1, uh1, bh1), (wz2, uz2, bz2, wr2, ur2, br2, wh2, uh2, bh2), v, c)

    hin1 = Tensor(he_normal((hd, 1))) if hin1 is None else hin1
    hin2 = Tensor(he_normal((hd, 1))) if hin2 is None else hin2
    loss = Tensor(0)

    for t in range(len(data) - 1) :
        hout1, hout2 = forward_pass(data[t], hin1, hin2, wub[:2])
        loss += cross_entropy(sigmoid(dense(v, hout2) + c), data[t + 1])

        if (t != 0) and (t % k == 0):
            loss.backward()

            for params in wub[:2] :
                for i, param in enumerate(params) :
                    rate = 1.
                    sgd(param, rate)
            for param in wub[2:] :
                rate = 1.
                sgd(param, rate)

            loss.null_gradients()

            loss = Tensor(0)
            wz1 = Tensor(wz1.data)
            uz1 = Tensor(uz1.data)
            bz1 = Tensor(bz1.data)
            wr1 = Tensor(wr1.data)
            ur1 = Tensor(ur1.data)
            br1 = Tensor(br1.data)
            wh1 = Tensor(wh1.data)
            uh1 = Tensor(uh1.data)
            bh1 = Tensor(bh1.data)
            wz2 = Tensor(wz2.data)
            uz2 = Tensor(uz2.data)
            bz2 = Tensor(bz2.data)
            wr2 = Tensor(wr2.data)
            ur2 = Tensor(ur2.data)
            br2 = Tensor(br2.data)
            wh2 = Tensor(wh2.data)
            uh2 = Tensor(uh2.data)
            bh2 = Tensor(bh2.data)
            v = Tensor(v.data)
            c = Tensor(c.data)
            hin1 = Tensor(hout1.data)
            hin2 = Tensor(hout2.data)

            wub = ((wz1, uz1, bz1, wr1, ur1, br1, wh1, uh1, bh1), (wz2, uz2, bz2, wr2, ur2, br2, wh2, uh2, bh2), v, c)
        else :
            hin1 = hout1
            hin2 = hout2
    return wub, hout1, hout2


def test_model_arr(wub, hin1, hin2, time, k=5, hd=100) :
    """
    OLD MODEL - TESTS ON ARRAYS
    time in sec
    """
    ticks = 2 * time * 96
    wub1, wub2, v, c = wub
    x = he_normal((len(v), 1))
    song_arr = np.empty((ticks, len(v)))

    for tick in range(ticks) :
        hout1, hout2 = forward_pass(x, hin1.data, hin2.data, (wub1, wub2))
        song_arr[tick] = round_notes(sigmoid(dense(v, hout2) + c).data)

        hin1 = hout1
        hin2 = hout2
        x = song_arr[tick]

    return song_arr


def round_notes(tick_arr, tolerance=.1) :
    """
    tick_arr : ndarray
    tolerance : size of uncertainty allowed for rounding up (above 1-tolerance is rounded up)
    """
    return np.where(tick_arr >= 1 - tolerance, 1, 0)


def sgd(param, rate) :
    """
    Performs a gradient-descent update on the parameter.

    Parameters
    ----------
    param : mygrad.Tensor
        The parameter to be updated.

    rate : float
        The step size used in the update"""
    param.data -= rate * param.grad
    return None


def forward_pass(x, hin1, hin2, wub) :
    """
    Computes the forward pass of a step in the RNN

    Parameters
    -----------
    x :

    hin1 :

    hin2 :

    wub :
    """
    wub1, wub2 = wub
    wz1, uz1, bz1, wr1, ur1, br1, wh1, uh1, bh1 = wub1
    wz2, uz2, bz2, wr2, ur2, br2, wh2, uh2, bh2 = wub2

    # first layer
    zt1 = sigmoid(dense(uz1, x[:, np.newaxis]) + dense(wz1, hin1) + bz1)
    rt1 = sigmoid(dense(ur1, x[:, np.newaxis]) + dense(wr1, hin1) + br1)
    ht1 = tanh(dense(uh1, x[:, np.newaxis]) + dense(wh1, (hin1 * rt1)) + bh1)
    hout1 = (1 - zt1) * hin1 + zt1 * ht1

    # second layer
    zt2 = sigmoid(dense(uz2, hout1) + dense(wz2, hin2) + bz2)
    rt2 = sigmoid(dense(ur2, hout1) + dense(wr2, hin2) + br2)
    ht2 = tanh(dense(uh2, hout1) + dense(wh2, (hin2 * rt2)) + bh2)

    hout2 = (1 - zt2) * hin1 + zt2 * ht2

    return hout1, hout2


def cross_entropy(p_pred, p_true):
    """ Computes the mean cross-entropy.

        Parameters
        ----------
        p_pred : mygrad.Tensor, shape:(N, K)
            N predicted distributions, each over K classes.

        p_true : mygrad.Tensor, shape:(N, K)
            N 'true' distributions, each over K classes

        Returns
        -------
        mygrad.Tensor, shape=()
            The mean cross entropy (scalar)."""
    # return (-(1 - p_true) * log(1 - p_pred) - p_true * log(p_pred)).sum()
    N = p_pred.shape[0]
    p_logq = (p_true) * log(p_pred)
    return (-1 / N) * p_logq.sum()


def he_normal(shape):
    """
    Given the desired shape of your array, draws random
        values from a scaled-Gaussian distribution.

    Returns
    --------
    numpy.ndarray
    """
    N = shape[0]
    scale = 1 / np.sqrt(2 * N)
    return np.random.randn(*shape) * scale


def import_midi(filename) :
    """
    Converts midi file to ndarray of shape (num_ticks, num_notes)
        0 for note off; 1 for note on
    """
    midi_song = mido.MidiFile(filename)

    if midi_song.type != 1 :
        raise Exception("Midi file type not supported")
    if midi_song.ticks_per_beat != 96 :
        raise Exception("Number of ticks per beat must be 96")

    # assume standard 120 bpm
    # 128 is number of notes in midi, hence 127
    # row dimension calculates and rounds up the number of ticks in song
    array_song = np.empty((int(-(-2 * midi_song.length * midi_song.ticks_per_beat // 1)), 127, len(midi_song.tracks[1:])))
    tick = 0

    # list of all notes being activated at given tick
    current_notes = []
    for i, track in enumerate(midi_song.tracks[1:]) :
        for msg in track:
            if msg.type == 'set_tempo' and msg.tempo != 500000 :
                raise Exception("Tempo must be 120 BPM (500000 MSPB)")

            if not msg.is_meta :
                if msg.type == 'note_on' and msg.velocity != 0 :
                    current_notes.append(msg.note)
                    array_song[tick : tick + msg.time, current_notes, i] = 1
                    tick += msg.time
                elif msg.type == 'note_off' or msg.type == 'note_on':
                    array_song[tick : tick + msg.time, current_notes, i] = 1
                    tick += msg.time
                    current_notes.remove(msg.note)
                else :
                    continue

        if len(current_notes) != 0 :
            array_song[tick : , current_notes] = 1
            current_notes = []

    array_song = np.sum(array_song, -1)

    return array_song
