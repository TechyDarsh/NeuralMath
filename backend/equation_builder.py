def build_equation(recognized_symbols):
    """
    Combine recognized symbols into an equation string.
    Apply basic heuristics if necessary.
    """
    equation_chars = [sym['char'] for sym in recognized_symbols]
    
    # A simple reconstruction (just join the chars)
    equation_str = "".join(equation_chars)
    
    return equation_str
