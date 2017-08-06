from mido import Message, MidiFile, MidiTrack
import numpy as np
import mido
import os.path
import pickle

# last key is 'split' key (indicates when to move to next line)
keys = ["!", "$", "%", "&", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6",
        "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
        "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i",
        "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", " "]

rnn_data = []


def import_dir(directory) :
    """
    Imports all valid midi files in directory

    Parameters
    -----------
    directory: r"PATH"
        path to directory of midi files
    """
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mid"):
            import_midi(os.path.join(directory, filename))
        else:
            continue


def import_midi(filename) :
    """
    Converts midi file to ndarray of shape (num_ticks, num_notes)
        0 for note off; 1 for note on

    Parameters
    -----------
    filename: r"PATH"
        filepath of midi file

    Returns
    --------
    str_song: str
        string representation of song
    """
    midi_song = mido.MidiFile(filename)

    # skips midi file if doesn't meet requirements
    if midi_song.type != 1 :
        return
        # raise Exception("Midi file type not supported")
    if midi_song.ticks_per_beat != 96 :
        return
        # raise Exception("Number of ticks per beat must be 96")

    # assume standard 120 bpm
    # 128 is number of notes in midi, hence 127
    # row dimension calculates and rounds up the number of ticks in song
    array_song = np.empty((int(-(-2 * midi_song.length * midi_song.ticks_per_beat // 1)), 127, len(midi_song.tracks[1:])))
    tick = 0

    # list of all notes being activated at given tick
    current_notes = []
    for i, track in enumerate(midi_song.tracks[1:]) :
        for msg in track:
            # skips song if change in tempo; potentially confusing to RNN
            if msg.type == 'set_tempo' and msg.tempo != 500000 :
                return
                #raise Exception("Tempo must be 120 BPM (500000 MSPB)")

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

    # converts ndarray representation to string representation for RNN
    # slices out all non-piano notes of midi file
    for tick in array_song[:, 20:108] :
        indices = np.where(tick == 1)
        indices = indices[0]

        temp_ls = []
        for index in indices :
            temp_ls.append(keys[index])
        ls_song.append("".join(temp_ls))

    str_song = " ".join(ls_song)
    rnn_data.append(str_song)

    # returns string representation of song
    return str_song


def str_to_midi(str_song, dirname=None) :
    """
    Converts string of note representations to midi file

    Parameters
    -----------
    str_song: str
        full string representation of midi song
    dirname: r"PATH"
        path to directory for midi file to be saved to

    Returns
    --------
        mid: MidiFile
            Mido MidiFile object
    """
    arr_song = np.zeros((len(str_song.split(' ')), len(keys) - 1), dtype=int)
    current_notes = []
    tick = 0

    # converts string representation to ndarry representation
    for i, char in enumerate(str_song) :
        if tick + 1 >= len(str_song):
            break
        else:
            if char == ' ':
                if len(current_notes) != 0:
                    indices = np.array([keys.index(x) for x in current_notes])
                    arr_song[tick, indices] = 1
                    current_notes.clear()
                tick += 1
            elif char not in current_notes:
                current_notes.append(char)

    # creates midi file
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 96

    current_notes = np.where(arr_song[0] == 1)[0].tolist()
    prev_tick = 0

    # converts ndarray representation to midi file
    for i, msg in enumerate(current_notes):
        track.append(Message('note_on', note=(21 + current_notes.index(msg)), velocity=64, time=0 + i))

    for tick in range(1, len(arr_song)) :
        ticks = np.sum((arr_song[tick], arr_song[tick - 1]), axis=0)
        # need to check for if note is off or is on
        if not np.array_equal(arr_song[tick], arr_song[tick - 1]) & len(ticks[ticks > 0]) != 0:
            diff = np.where(arr_song[tick] != arr_song[tick - 1])[0]
            for index in diff:
                # if note was added
                if arr_song[tick, index] == 1:
                    track.append(Message('note_on', note=(21 + index), velocity=64, time=0))
                # if note ended
                else:
                    track.append(Message('note_on', note=(21 + index), velocity=0, time=tick - prev_tick))
            prev_tick = tick

    current_notes = np.where(arr_song[len(arr_song) - 1] == 1)[0].tolist()
    for msg in current_notes:
        track.append(Message('note_on', note=(21 + current_notes.index(msg)), velocity=0, time=prev_tick))

    if dirname is not None :
        if os.path.exists(dirname):
            mid.save(os.path.join(dirname, 'generated_song.mid'))
    else :
        mid.save('generated_song.mid')

    return mid


def save() :
    '''
    Saves face_data to a .pickle file
    '''
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "music_gen.pickle"), 'wb') as f:
        pickle.dump("   ".join(rnn_data), f, pickle.HIGHEST_PROTOCOL)
