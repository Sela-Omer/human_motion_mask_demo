from typing import Union, List


def convert_param_to_type(s: str) -> Union[List[Union[int, float, str]], int, float, str]:
    """
    Convert a string to an integer, float, or list of integers, floats, or strings.
    :param s: The string to convert.
    :return: The converted value.
    """
    # Try to convert the string to an integer
    try:
        return int(s)
    # If conversion to integer fails, try to convert to float
    except ValueError:
        try:
            return float(s)
        # If conversion to float fails, return the string as is
        except ValueError:
            lst = s.split(',')
            if len(lst) > 1:
                return [s for s in lst]
            return s


def convert_param_to_list(s: str) -> List[str]:
    """
    Convert a string to a list of str.
    :param s: The string to convert.
    :return: The converted list.
    """
    if s == '':
        return []
    if ',' not in s:
        return [s]
    return s.split(',')