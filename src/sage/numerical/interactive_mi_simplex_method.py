from sage.numerical.interactive_simplex_method import *
from sage.plot.all import Graphics, arrow, line, point, rainbow, text
from sage.rings.all import Infinity, PolynomialRing, QQ, RDF, ZZ

class InteractiveMILPProblem(InteractiveLPProblem):

    def __init__(self, A, b, c, x="x",
                 constraint_type="<=", variable_type="", 
                 problem_type="max", base_ring=None, 
                 is_primal=True, objective_constant_term=0, 
                 integer_variables=False):
        super(InteractiveMILPProblem, self).__init__(A, b, c, x="x",
                 constraint_type="<=", variable_type="", 
                 problem_type="max", base_ring=None, 
                 is_primal=True, objective_constant_term=0)
        R = PolynomialRing(self.base_ring(), list(self.Abcx()[3]), order="neglex")
        if integer_variables is False:
            self._integer_variables = set([])
        elif integer_variables is True:
            self._integer_variables = set(self._Abcx[3])
        else:
            self._integer_variables = set([variable(R, v)
                                           for v in integer_variables])

    def add_constraint(self, coefficients, new_b, 
                            new_constraint_type="<=", integer_slack=False):
        r"""
        Return a new MILP problem by adding a constraint to``self``.

        INPUT:

        - ``coefficients`` -- coefficients of the new constraint

        - ``new_b`` -- a constant term of the new constraint

        - ``new_constraint_type`` -- (default: ``"<="``) a string indicating
          the constraint type of the new constraint

        OUTPUT:

        - an :class:`MILP problem <InteractiveLPProblem>`

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c)
            sage: P1 = P.add_constraint(([2, 4]), 2000, new_constraint_type="<=")
            sage: P1.Abcx()
            (
            [1 1]
            [3 1]
            [2 4], (1000, 1500, 2000), (10, 5), (x1, x2)
            )
            sage: P1.constraint_types()
            ('<=', '<=', '<=')
            sage: P.Abcx()
            (
            [1 1]
            [3 1], (1000, 1500), (10, 5), (x1, x2)
            )
            sage: P.constraint_types()
            ('<=', '<=')
            sage: P2 = P.add_constraint(([2, 4, 6]), 2000, new_constraint_type="<=")
            Traceback (most recent call last):
            ...
            ValueError: A and coefficients have incompatible dimensions
            sage: P3 = P.add_constraint(([2, 4]), 2000, new_constraint_type="<")
            Traceback (most recent call last):
            ...
            ValueError: unknown constraint type
        """
        if self.n_variables() != matrix(coefficients).ncols():
            raise ValueError("A and coefficients have incompatible dimensions")
        if new_constraint_type in ["<=", ">=", "=="]:
            constraint_type = self._constraint_types + (new_constraint_type,)
        else:
            raise ValueError("unknown constraint type")
        A = self.Abcx()[0]
        b = self.Abcx()[1]
        c = self.Abcx()[2]
        A = A.stack(matrix(coefficients))
        b = vector(tuple(b) + (new_b,))
        if self._is_negative:
            problem_type = "-" + self.problem_type()
        else:
            problem_type = self.problem_type()
        return InteractiveMILPProblem(A, b, c, x=self.Abcx()[3],
                    constraint_type=constraint_type,
                    variable_type=self.variable_types(),
                    problem_type=problem_type,
                    objective_constant_term=self.objective_constant_term(),
                    integer_variables=self.integer_variables())

    def all_variables(self):
        r"""
        Return a set of all decision variables of ``self``.

        OUTPUT:

        - a set of variables

        EXAMPLES::

            sage: A = ([1, 2, 1], [3, 1, 5])
            sage: b = (1000, 1500)
            sage: c = (10, 5, 7)
            sage: P = InteractiveMILPProblem(A, b, c)
            sage: P.all_variables()
            {x1, x2, x3}
        """
        return set(self.Abcx()[3])

    def continuous_variables(self):
        r"""
        Return a set of continuous decision variables of ``self``.

        OUTPUT:

        - a set of variables

        EXAMPLES::

            sage: A = ([1, 2, 1], [3, 1, 5])
            sage: b = (1000, 1500)
            sage: c = (10, 5, 7)
            sage: P = InteractiveMILPProblem(A, b, c, integer_variables={'x1'})
            sage: P.continuous_variables()
            {x2, x3}
        """
        I = self.integer_variables()
        all_variables = self.all_variables()
        C = all_variables.difference(I)
        return C

    def integer_variables(self):
        r"""
        Return the set of integer decision variables of ``self``.

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1/10, 15/10)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, integer_variables={'x1'})
            sage: P.integer_variables()
            {x1}
            sage: P = InteractiveMILPProblem(A, b, c, integer_variables=True)
            sage: P.integer_variables()
            {x1, x2}
            sage: P = InteractiveMILPProblem(A, b, c, integer_variables=False)
            sage: P.integer_variables()
            set()
        """
        return self._integer_variables