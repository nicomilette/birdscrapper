import os

def main_menu():
    print("Select an option:")
    print("1. Scrape and Filter Data")
    print("2. Statistics - Optional")
    print("3. Download Recordings")
    print("4. Spectrogram Selection")
    print("5. Preprocess Data")
    print("6. Verify Files")
    print("7. Train and Save Model")
    print("8. Test Predict")
    print("0. Exit")

def execute_script(script_name):
    command = f"python {script_name}.py"
    os.system(command)

if __name__ == "__main__":
    while True:
        main_menu()
        choice = input("Enter your choice (0-8): ")

        if choice == '1':
            execute_script('src/scrape')
            execute_script('src/filter')
        elif choice == '2':
            choice1 = input("1. Generate heatmap\n2. Generate recording statistics\n")
            if choice1 == '1':
                execute_script('src/heatmap')
            elif choice1 == '2':
                execute_script('src/stats')
            else:
                print('Invalid choice.')
        elif choice == '3':
            execute_script('src/download')
        elif choice == '4':
            execute_script('webapp/app')
        elif choice == '5':
            execute_script('src/preprocess')
        elif choice == '6':
            execute_script("src/verify")
        elif choice == '7':
            execute_script("src/train")
        elif choice == '8':
            execute_script("src/predict")
        elif choice == '0':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")
