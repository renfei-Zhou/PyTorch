import requests
from pathlib import Path

# Download web function
file_name = "helper_functions.py"
file_web_dir = "http://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py"


if Path(file_name).is_file():
    print(f"{file_name} already exists, skip downloading")
else:
    print(f"Downloading {file_name}")
    request = requests.get(file_web_dir)
    with open(file_name, "wb") as f:
        f.write(request.content)


