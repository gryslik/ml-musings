import Mario_Trainer
def main():
    response = None
    response = input("Welcome to the mario trainer, what would you like to do? Train new models [1], Run all existing models [2], record a specific model [3], or restart training [4]: ")
    while response != "1" and response != "2" and response != "3" and response != "4":
        response = input("Please enter either 1, 2, 3 , or 4 as your answer: ")
    
    if response == "1":
        Mario_Trainer.train_models()
    elif response == "2":
        Mario_Trainer.run_all_models()
    elif response == "3":
        model_number = input("Enter the model number you wish to record: ")
        Mario_Trainer.record_video(model_number)
    else:
        model_number = input("Enter the model you wish to restart training from: ")
        epsilon = input("Enter the epsilon value from the training log for this model: ")
        Mario_Trainer.train_models(model_number, epsilon)


main()