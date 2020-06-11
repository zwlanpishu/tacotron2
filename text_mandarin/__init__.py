from text_mandarin.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"


def text_to_sequence(text):
    """
    A string convert to a list of index
    """
    sequence = _symbols_to_sequence(text)
    sequence.append(_symbol_to_id['~'])
    return sequence


def sequence_to_text(sequence):
    """
    A list of index convert to a string
    """
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol and symbol_id != 0 and symbol_id != 1:
            s = _id_to_symbol[symbol_id]
            result += s
    return result
