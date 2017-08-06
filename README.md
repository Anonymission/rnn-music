# RNN Music
This is a student group project for MIT Beaver Works Summer Institute 2017: Cognitive Assistant Collaboration.

The project is designed to convert midi files to a trainable string format for a RNN, or other LSTM variation.

### Setup.py
Install these packages and follow the instructions as stated:
[Mido](http://mido.readthedocs.io/en/latest/installing.html)

Clone this repository.
Open the folder of the cloned repository in Command Prompt.
Enter this command:
```shell
python setup.py develop
```

# Training a model
The code in this repository simply converts midi files to strings, which are more suitable for training in a RNN. The examples in the Music folder in this repository were trained using [Andrej Karptahy's char-rnn](https://github.com/karpathy/char-rnn).
