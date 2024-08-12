import subprocess, os

envs = os.listdir('C:\\dev\\Python\\venvs\\')
envs_d = {i: e for i, e in enumerate(envs)}
print(f"[c] create, {envs_d}")
choice = input("Enter your choice: ")

if choice == "c":
    venv = input("Enter new venv name: ")
    subprocess.run(f"virtualenv -p C:\\dev\\Python\\Python310\\python.exe C:\\dev\\Python\\venvs\\{venv}".split())
elif choice.isdigit():
    c = f"C:\\dev\\Python\\venvs\\{envs_d[int(choice)]}\\Scripts\\activate"
    print(f'{c=}')
    os.system(f'cmd /k "{c}"')

else:
    print("Invalid choice.")
    input("Press Enter to continue...")

subprocess.run(["python", "-V"], shell=True)
input("Press Enter to continue...")