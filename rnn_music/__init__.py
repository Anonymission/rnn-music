from mygrad import Tensor
from mygrad.nnet.activations import sigmoid, tanh, softmax
from mygrad.nnet.layers import dense
from mygrad.math import log
import numpy as np
import mido
import os.path
import pickle

# last ky is 'split' key (indicates when to move to next line)
keys = ["!", "$", "%", "&", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6",
        "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
        "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i",
        "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", " "]

rnn_data = []
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "music_gen.pickle"), 'rb') as f:
    rnn_data = pickle.load(f)


def train_model(data, k=5, hd=700, wub=None, hin1=None, hin2=None) :
    """
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
    # initialize weights
    md = len(keys)
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

    for i in range(len(data) - 1) :
        x = one_hot_encode(data, i)

        hout1, hout2 = forward_pass(x, hin1, hin2, wub[:2])

        x_next = one_hot_encode(data, i + 1)

        loss += cross_entropy(softmax(dense(v, hout2) + c), x_next)
        loss.backward()
        loss.null_creators()

        if (i != 0) and (i % k == 0):
            print("reset")
            for params in wub[:2] :
                for param in params :
                    rate = 1.
                    sgd(param, rate)
            for param in wub[2:] :
                rate = 1.
                sgd(param, rate)

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
            x = x_next
        else :
            hin1 = hout1
            hin2 = hout2
            x = x_next

        if i % 1000 == 0 :
            rnn_data.append([wub, hin1, hin2, i])
            save()
    rnn_data.append([None])
    return wub, hout1, hout2


def test_model(wub, hin1, hin2, length, k=5, hd=100) :
    """
    length of chars
    """
    ls = []
    x = one_hot_encode()
    while len(ls) < length :
        hout1, hout2 = forward_pass(x, hin1, hin2, wub[:2])

        # working softmax
        d = np.dot(wub[2].data, hout2.data) + wub[3].data
        np.exp(d, out=d)
        sm = d / np.sum(d)
        index = np.argmax(sm)

        # tensor softmax not working, no idea why
        # index = np.argmax(softmax(dense(wub[2], hout2) + wub[3]).data)

        ls.append(keys[index])

        x = one_hot_encode(ls[-1], 0)
        hin1 = hout1
        hin2 = hout2

    return ls


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
    zt1 = sigmoid(dense(uz1, x) + dense(wz1, hin1) + bz1)
    rt1 = sigmoid(dense(ur1, x) + dense(wr1, hin1) + br1)
    ht1 = tanh(dense(uh1, x) + dense(wh1, (hin1 * rt1)) + bh1)
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


def one_hot_encode(data=None, i=None) :
    """
    optional params. both must be filled if one is. if neither, one hot encode a space
    """
    if (data is not None and i is None) or (data is None and i is not None) :
        raise Exception("Either both data and i must be passed values, or neither are passed values")
    elif data is not None and i is not None :
        x = np.zeros((len(keys), 1))
        x[keys.index(data[i])] = 1
    else :
        x = np.zeros((len(keys), 1))
        x[-1] = 1
    return x


def import_midi(filename, sr) :
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

    ls_song = []

    for tick in array_song[::sr,21:109] :
        indices = np.where(tick == 1)
        indices = indices[0]

        temp_ls = []
        for index in indices :
            temp_ls.append(keys[index])
        ls_song.append("".join(temp_ls))

    str_song = " ".join(ls_song)

    return array_song, ls_song, str_song

'''
def str_to_arr(str_song) :
    """
    data = str
    """
    arr_song = np.zeros((len(str_song.split()), len(keys) - 1))
    tick = 0
    for char in data :
        if char == keys[-1]
            tick += 1
        else :
            arr_song[tick, keys.index(char)] = 1

    return arr_song


def arr_to_midi(arr_song) :
    """
    arr_song = ndarray
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    current_notes = np.where(arr_song[0] == 1)[0].tolist()
    # track.append(Message('note_on', note=64, velocity=64, time=32))
    for tick in range(1, len(arr_song)) :
        if arr_song[tick] != arr_song[tick - 1] :

        else :
            continue
'''

def save() :
    '''
    Saves face_data to a .pickle file
    '''
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "music_gen.pickle"), 'wb') as f:
        pickle.dump(rnn_data, f, pickle.HIGHEST_PROTOCOL)
