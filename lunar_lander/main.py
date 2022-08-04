import eagle_large
import os
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

valid_options = {"1", "2"}
response = ""
while response not in valid_options:
    response = input("Welcome to the Lunar Lander trainer, what would you like to do?\n[1] Train the agent\n[2] Record an existing model\nPlease enter 1 or 2: ")

    if response == "1":
        eagle_large.train_agent()
    elif response == "2":
        file_path = input("Enter the ABSOLUTE file path for the model: ")
        while not os.path.exists(file_path) or pathlib.Path(file_path).suffix != ".h5":
            print("What you entered was not a valid model file path. Please make sure the file exists and is a .h5 file")
            file_path = input("Enter the ABSOLUTE file path for the model: ")
        eagle_large.record_model(file_path, True)
    else:
        print("Please enter \"1\" or \"2\": ")

