import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

def solve_equation(equation_str):
    """
    Use SymPy to solve the equation.
    Handles basic equations (e.g. 2*x + 5 = 9 or 2x + 5 = 9).
    """
    try:
        # Define transformations to handle things like '2x' -> '2*x'
        transformations = (standard_transformations + (implicit_multiplication_application,))
        
        if '=' in equation_str:
            lhs_str, rhs_str = equation_str.split('=', 1)
            
            lhs = parse_expr(lhs_str, transformations=transformations)
            rhs = parse_expr(rhs_str, transformations=transformations)
            
            equation = sp.Eq(lhs, rhs)
            
            # Find the variables in the equation
            symbols = list(equation.free_symbols)
            if len(symbols) == 0:
                # E.g., 5 = 5 or 5 = 9
                outcome = bool(lhs == rhs)
                return str(outcome)
            elif len(symbols) == 1:
                # Solve for the single variable
                var = symbols[0]
                solutions = sp.solve(equation, var)
                if len(solutions) == 1:
                    return f"{var} = {solutions[0]}"
                else:
                    return f"{var} = {solutions}"
            else:
                # Multivariable
                try:
                    solutions = sp.solve(equation, symbols)
                    return str(solutions)
                except:
                    return "Unable to resolve multivariable equation"
        else:
            # Not an equation with '=', maybe just an expression to evaluate
            expr = parse_expr(equation_str, transformations=transformations)
            return str(sp.simplify(expr))
            
    except Exception as e:
        return f"Error solving equation: {str(e)}"
