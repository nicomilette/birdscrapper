import os

def main_menu():
    print("Select an option:")
    print("1. Scrape Data")
    print("2. Generate Heatmap - Optional")
    print("3. Generate Recording Statistics - Optional")
    print("4. Filter Recordings")
    print("5. Download Recordings")
    print("6. Preprocess Data")
    print("7. Verify Files")
    print("8. Train and Save Model")
    print("9. Test Predict")
    print("0. Exit")

def execute_script(script_name):
    command = f"python {script_name}.py"
    os.system(command)

if __name__ == "__main__":
    while True:
        main_menu()
        choice = input("Enter your choice (0-9): ")

        if choice == '1':
            execute_script('src/scrape')
        elif choice == '2':
            execute_script('src/heatmap')
        elif choice == '3':
            execute_script('src/stats')
        elif choice == '4':
            execute_script('src/filter')
        elif choice == '5':
            execute_script('src/download')
        elif choice == '6':
            execute_script('src/preprocess')
        elif choice == '7':
            execute_script("src/verify")
        elif choice == '8':
            execute_script("src/train")
        elif choice == '9':
            execute_script("src/predict")
        elif choice == '0':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")
