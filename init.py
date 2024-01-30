import subprocess

# List of pip install commands
pip_commands = [
    "pip install Flask",
    "pip install nltk",
    "pip install rouge-score",
    "pip install matplotlib",
    "pip install sacrebleu",
    "pip install pdfminer.six"
]

# Run each pip command
for cmd in pip_commands:
    subprocess.run(cmd, shell=True)
