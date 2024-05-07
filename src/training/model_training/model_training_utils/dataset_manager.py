import pandas as pd
import numpy as np
from sklearn import model_selection
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Gets TensorFlow to shut up...

import tensorflow as tf

class ChordsDatasetManager:
    def __init__(self, dataset_path: str) -> None:
        
        """
        A class that takes in the path to a dataset, and does all the necessary processing
        on it, such as making the input and output columns, splitting the test and train
        datasets, and returning tf.data.Dataset objects for the LSTM model.

        Parameters:
            dataset_path: str      A relative or absolute path to the dataset CSV file.
        """

        # The columns expected to be found in the CSV file
        self.__EXPECTED_CSV_COLUMNS = ["melody_chroma", "0", "1", "2", "3", "4",
                                     "5", "6", "7", "8", "9", "10", "11"]

        # Load the dataset CSV file
        self.__DATASET_PATH = dataset_path
        self.__dataset = pd.read_csv(self.__DATASET_PATH)
        print("Loaded dataset from file...")
        
        # Remove rows containing NaN values
        self.__dataset.dropna(inplace=True)

        # Check that all the expected columns are in the loaded CSV file
        for column_name in self.__dataset.columns.to_list():
            if column_name not in self.__EXPECTED_CSV_COLUMNS:
                raise ValueError(f"Incorrect columns in loaded data frame. Expected {self.__EXPECTED_CSV_COLUMNS}, but found {self.__dataset.columns.to_list()}.")

        # Create a copy to be modified by later processes
        self.__formatted_dataset = self.__dataset.copy()
        
        self.__is_formatted = False
        self.__is_test_train_split = False

        # Define new input column names
        self.__NEW_INPUT_COLUMN_NAMES = {"0":"input_0", "1":"input_1", "2":"input_2",
                            "3":"input_3", "4":"input_4", "5":"input_5",
                            "6":"input_6", "7":"input_7", "8":"input_8",
                            "9":"input_9", "10":"input_10", "11":"input_11"}

        # Define new output column names
        self.__NEW_OUTPUT_COLUMN_NAMES = {"0":"output_0", "1":"output_1", "2":"output_2",
                            "3":"output_3", "4":"output_4", "5":"output_5",
                            "6":"output_6", "7":"output_7", "8":"output_8",
                            "9":"output_9", "10":"output_10", "11":"output_11"}
    
    def format_dataset(self) -> None:

        if not self.__is_formatted:

            # Get input columns, shift down by one, rename
            input_chroma_columns = self.__formatted_dataset[list(self.__NEW_INPUT_COLUMN_NAMES.keys())]
            input_chroma_columns = input_chroma_columns.shift(periods=1, fill_value=0)
            input_chroma_columns.rename(columns=self.__NEW_INPUT_COLUMN_NAMES, inplace=True)

            # Rename chroma columns to be outputs
            self.__formatted_dataset.rename(columns=self.__NEW_OUTPUT_COLUMN_NAMES, inplace=True)

            # Add input columns onto main formatted dataframe
            self.__formatted_dataset = pd.concat((self.__formatted_dataset, input_chroma_columns), axis=1)

            # Rename melody_chroma column to input_melody_chroma
            self.__formatted_dataset.rename(columns={"melody_chroma":"input_melody_chroma"}, inplace=True)
            
            # Progress update
            print("Shifted input columns...")

            input_cols = ["input_melody_chroma",
                        "input_0", "input_1", "input_2",
                        "input_3", "input_4", "input_5",
                        "input_6", "input_7", "input_8",
                        "input_9", "input_10", "input_11"]

            output_cols = ["output_0", "output_1", "output_2",
                        "output_3", "output_4", "output_5",
                        "output_6", "output_7", "output_8",
                        "output_9", "output_10", "output_11"]

            self.__input_data = self.__formatted_dataset[input_cols].to_numpy()
            self.__output_data = self.__formatted_dataset[output_cols].to_numpy()

            print("Generated input and output data arrays...")
            print(f"Input data shape:  {self.__input_data.shape}")
            print(f"Output data shape: {self.__output_data.shape}")

            self.__is_formatted = True
            print("---")
            print("Completed formatting dataset")
            print("------")

        else:
            print("Dataset already formatted, bypassing...")

    def test_train_split(
            self,
            test_size: float = None,
            sequence_length: int = 8,
            batch_size: int = 64
        ) -> None:
        
        # Splitting the dataset into training and testing
        self.__input_data_train, self.__input_data_test, self.__output_data_train, self.__output_data_test = model_selection.train_test_split(self.__input_data, self.__output_data, test_size=test_size)

        print("Split dataset into training & testing data...")
        print("---")
        print(f"Input train shape:\t{self.__input_data_train.shape}")
        print(f"Input test shape:\t{self.__input_data_test.shape}")
        print(f"Output train shape:\t{self.__output_data_train.shape}")
        print(f"Output test shape:\t{self.__output_data_test.shape}")
        print("------")

        # Convert to tf.data.Dataset objects
        self.__dataset_train = tf.keras.utils.timeseries_dataset_from_array(
            data=self.__input_data_train,
            targets=self.__output_data_train,
            sequence_length=sequence_length,
            batch_size=batch_size
        )

        self.__dataset_test = tf.keras.utils.timeseries_dataset_from_array(
            data=self.__input_data_test,
            targets=self.__output_data_test,
            sequence_length=sequence_length,
            batch_size=batch_size
        )

        # Update boolean
        self.__is_test_train_split = True

    def get_training_data(self) -> tf.data.Dataset:
        if self.__is_test_train_split:
            return self.__dataset_train
        else:
            raise ValueError(f"Dataset has not been split into training and testing data. Must run DatasetManager.format() and DatasetManager.test_train_split() first.")
        
    def get_test_data(self) -> tf.data.Dataset:
        if self.__is_test_train_split:
            return self.__dataset_test
        else:
            raise ValueError(f"Dataset has not been split into training and testing data. Must run DatasetManager.format_dataset() and DatasetManager.test_train_split() first.")
        
    def get_raw_dataset(self) -> pd.DataFrame:
        return self.__dataset
    
    def get_output_data_test(self) -> np.ndarray:
        return self.__output_data_test

if __name__ == "__main__":

    dataset_manager = ChordsDatasetManager("C:/Users/jacke/OneDrive - Universitetet i Oslo/Thesis/thesis_repo/chord_generation/model_training/chords_datasets/Lakh_2024/Lakh_chords_dataset_2024_ALL.csv")
    dataset_manager.format_dataset()
    dataset_manager.test_train_split(test_size=0.3, sequence_length=8, batch_size=64)
    dataset_manager.get_training_data()
    dataset_manager.get_test_data()
    dataset_manager.get_output_data_test()