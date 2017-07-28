from mygrad import Tensor
from mygrad.nnet.activations import sigmoid, tanh, softmax
from mygrad.nnet.layers import dense
from mygrad.math import log
import numpy as np
import mido


def train_model(data, k=6) :
    """
    Trains model of RNN

    Parameters
    -----------
    data: ndarray
        array of data (what form?)
    k: int; default = 3
        number of backprops to undergo
    """
    wz = Tensor(he_normal((len(data[0]), 100)))
    wr = Tensor(he_normal((len(data[0]), 100)))
    wh = Tensor(he_normal((len(data[0]), 100)))
    hout = Tensor(np.zeros((len(data[0]), 100), dtype=wh.dtype))
    loss = Tensor(0)
    hin = Tensor(np.zeros((len(data[0]), 100), dtype=wh.dtype))

    for t in range(len(data) - 1) :
        wb = (wz, wr, wh)
        hout, C = forward_pass(data[t], hin, wb)
        pred = softmax(hout)
        loss += cross_entropy(pred, data[t + 1])

        if (t != 0) and (t % k == 0):
            loss.backward()

            for param in wb :
                rate = 1.
                sgd(param, rate)

            loss.null_gradients()

            hin = Tensor(hout.data)
        else :
            hin = hout

    return wb, hout


def sgd(param, rate) :
    """ Performs a gradient-descent update on the parameter.

        Parameters
        ----------
        param : mygrad.Tensor
            The parameter to be updated.

        rate : float
            The step size used in the update"""
    param.data -= rate * param.grad
    return None


def forward_pass(x, hin, wb) :
    """
    Computes the forward pass of a step in the RNN

    Parameters
    -----------
    x:

    C:
    """
    inputs = np.vstack((x, hin))
    wz, wr, wh = wb
    rt = sigmoid(dense(wr, inputs))
    zt = sigmoid(dense(wz, inputs))
    ht = tanh(dense(wh, np.vstack(x, hin * rt)))
    hout = (1 - zt) * hin + zt * ht

    return hout


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

    N = p_pred.shape[0]
    p_logq = (p_true) * log(p_pred)
    return (-1 / N) * p_logq.sum()


def he_normal(shape):
    """ Given the desired shape of your array, draws random
        values from a scaled-Gaussian distribution.

        Returns
        -------
        numpy.ndarray"""
    N = shape[0]
    scale = 1 / np.sqrt(2 * N)
    return np.random.randn(*shape) * scale


def import_midi(filename) :
    """
    """
    midi_song = mido.MidiFile(filename)

    if midi_song.type != 1 :
        raise Exception("Midi file type not supported")
    if midi_song.ticks_per_beat != 96 :
        raise Exception("Number of ticks per beat must be 96")

    # assume standard 120 bpm
    # 128 is number of notes in midi, hence 127
    array_song = np.empty(((int(np.ceil(2 * midi_song.length * midi_song.ticks_per_beat)), 127, len(midi_song.tracks[1:]))))
    tick = 0

    # list of all notes being activated at given tick
    current_notes = []
    for i, track in enumerate(midi_song.tracks[1:]) :
        for msg in track:
            if msg.type == 'set_tempo' and msg.tempo != 500000 :
                raise Exception("Tempo must be 120 BPM (500000 MSPB)")

            if not msg.is_meta :
                if msg.type == 'note_on' and msg.velocity != 0:
                    current_notes.append(msg.note)
                elif msg.type == 'note_off' or msg.type == 'note_on':
                    current_notes.remove(msg.note)
                else :
                    continue
                array_song[tick : tick + msg.time, current_notes, i] = 1
            tick += msg.time

        if len(current_notes) != 0 :
            array_song[tick : , current_notes] = 1
            current_notes = []

    array_song = np.sum(array_song, -1)
    array_song[np.where(array_song > 1)] = 1
    return array_song
