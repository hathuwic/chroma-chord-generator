import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from .layers import CustomLSTMLayer1, CustomLSTMLayer2, CustomLSTMLayer3, OutputLayer

class ChordGeneratorModel(Model):
    def __init__(
            self,
            lstm1_units: int,
            lstm2_units: int,
            lstm3_units: int,
            num_inputs: int = 13,
            num_outputs: int = 12,
            dropout = 0.25
        ) -> None:

        super(ChordGeneratorModel, self).__init__()
        
        self.__lstm1 = CustomLSTMLayer1(
            lstm1_units,
            num_inputs,
            dropout=dropout
        )

        self.__lstm2 = CustomLSTMLayer2(
            lstm2_units,
            dropout=dropout
        )

        self.__lstm3 = CustomLSTMLayer3(
            lstm3_units,
            dropout=dropout
        )

        self.__output_layer = OutputLayer(
            num_outputs
        )

    def call(self, inputs):
        x = self.__lstm1(inputs)
        x = self.__lstm2(x)
        x = self.__lstm3(x)
        x = self.__output_layer(x)
        return x
    
if __name__ == "__main__":

    # Test creating, compiling, and building untrained model

    lstm1_units = 512
    lstm2_units = 512
    lstm3_units = 256
    num_inputs = 13
    num_outputs = 12
    sequence_length = 8

    learning_rate = 0.0001

    dropout = 0.375

    input_shape = (None, sequence_length, num_inputs)

    model = ChordGeneratorModel(lstm1_units, lstm2_units, lstm3_units, num_outputs=num_outputs, dropout=dropout)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredLogarithmicError()

    model.compile(
        optimizer=optimizer,
        loss=loss
    )

    model.build(input_shape=input_shape)
    model.summary()