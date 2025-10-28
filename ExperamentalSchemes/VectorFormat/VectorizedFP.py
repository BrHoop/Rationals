import numpy as np
SIZE = 10
DEC_POS = 5


def convert_to_vector(x):
    digits = str(x)
    nondec, dec = digits.split(".")
    nondec = to_vec(nondec, SIZE - DEC_POS)
    dec = to_vec(dec, DEC_POS, True)
    nondec = np.array(nondec)
    dec = np.array(dec)
    result = np.concatenate((nondec, dec))

    return result


def to_vec(val: str, sz, isDec=False) -> list[int]:
    pairs = [val[i:i + 2][::-1] for i in range(0, len(val), 2)]
    result = [int(p[::-1]) for p in pairs]
    if isDec:
        result.extend([0] * (sz - len(result)))
    else:
        result = [0] * (sz - len(result)) +result
    return result


def main ():

    x = 1321.4 #Test number
    y = 5646.1 # Test number

    vect_x = np.array(convert_to_vector(x))
    vect_y = np.array(convert_to_vector(y))
    print(vect_x)
    print(vect_y)

    print(vect_x + vect_y)
    print(vect_x * vect_y)



if __name__ == "__main__":
    main()