import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, LSTM, Dense

class CustomLSTMLayer1(Layer):
    def __init__(
            self,
            units: int,
            num_inputs: int,
            dropout: float
        ) -> None:
        
        super(CustomLSTMLayer1, self).__init__()
        
        self.lstm = LSTM(
            units,
            return_sequences=True,
            input_shape=(None, num_inputs),
            dropout=dropout
        )

    def call(self, inputs):
        return self.lstm(inputs)
    
class CustomLSTMLayer2(Layer):
    def __init__(
            self,
            units: int,
            dropout: float
        ) -> None:

        super(CustomLSTMLayer2, self).__init__()

        self.lstm = LSTM(
            units,
            return_sequences=True,
            dropout=dropout
        )

    def call(self, inputs):
        return self.lstm(inputs)

class CustomLSTMLayer3(Layer):
    def __init__(
            self,
            units: int,
            dropout: float
        ) -> None:

        super(CustomLSTMLayer3, self).__init__()

        self.lstm = LSTM(
            units,
            return_sequences=False,
            dropout=dropout
        )

    def call(self, inputs):
        return self.lstm(inputs)
    
class OutputLayer(Layer):
    def __init__(
            self,
            num_outputs
        ) -> None:

        super(OutputLayer, self).__init__()

        self.dense = Dense(num_outputs)

    def call(self, inputs):
        return self.dense(inputs)