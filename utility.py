import os


# Clear the terminal
def clear():
    os.system("cls" if os.name == "nt" else "clear")


def press_enter():
    input("\nPress Enter to continue...")
