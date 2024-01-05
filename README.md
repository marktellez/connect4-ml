# Connect 4 ML

Howdy. This uses a mixed model (CNN, FF) to learn the board and features generated from the gui game (bot vs bot).

## Generating game files (training data)

```python gui/main.py --bot_vs_bot --num_games 10000```

## Training the Model

```python model/train.py```

The model will train on a number of hyperparms and each permutation will early stop if the training isn't showing the model improving its ability to predict good moves and therefore increase accuracy.

Graphs are stored at each epoch. Models are stored at every increase in accuracy.

## Playing against your model


```python gui/main.py --num_games 10 --model_path models/connect4_model_16_0.001_4_4_8```

### Author

Mark Tellez
Sr. Programmer