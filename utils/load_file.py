import io
import json
import os


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def load_json_file(path: str):
    try:
        list_data_dict = jload(path)
    except BaseException:
        with open(path, "r") as f:
            lines = f.readlines()
        list_data_dict = []
        for line in lines:
            try:
                json_str = json.loads(line.strip())
                list_data_dict.append(json_str)
            except Exception:
                print("--------------------")
                print(line)
                print("--------------------")
    return list_data_dict


def find_files_with_suffix(directory, suffix=".bin"):
    bin_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                bin_files.append(os.path.join(root, file))
    return bin_files
