import enum as _enum
import pickle


def enum2tensor(e):
    tensor = list()
    for i in type(e):
        if i is e:
            tensor.append(1)
        else:
            tensor.append(0)
    return tensor


def flatten_tensor(tensor):
    if isinstance(tensor, list):
        result = list()
        for item in tensor:
            result += flatten_tensor(item)
        return result
    elif isinstance(tensor, _enum.Enum):
        return enum2tensor(tensor)
    else:
        return tensor


def load_pickle(file_path):
    objs = []
    f = open(file_path, 'rb')
    while 1:
        try:
            objs.append(pickle.load(f))
        except EOFError:
            break
    return objs
