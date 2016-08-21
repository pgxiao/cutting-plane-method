from copy import copy
from sage.misc.html import HtmlFragment
from sage.numerical.interactive_simplex_method import *
from sage.numerical.interactive_simplex_method import _latex_product
from sage.plot.all import Graphics, arrow, line, point, rainbow, text
from sage.rings.all import Infinity, PolynomialRing, QQ, RDF, ZZ

def _form_thin_long_triangle(k):
    r"""
    Generate a thin long triangle.

    .. NOTE::

        :meth:`_form_thin_long_triangle` is for internal use. Generate a
        thin long triangle with vertices `(0, 0)`, `(1, 0)`, and `(1/2, k)`
        for some given integer `k`, and return a matrix `A`, and an vector
        `b`, where the triangle is represented by a polytope defined by
        `Ax <= b`. This thin long triangle is an example of a system with
        large Chvatal rank.

    INPUT:

    - ``k``-- an integer indicating the y coordinate of the top vertex
      for the triangle

    OUTPUT:

    - ``A`` -- a two by two matrix

    - ``b`` -- a two-element vector

    EXAMPLES::

        sage: from sage.numerical.interactive_milp_problem \
        ....:     import _form_thin_long_triangle
        sage: A, b, = _form_thin_long_triangle(4)
        sage: A, b
        (
            [-8  1]
            [ 8  1], (0, 8)
            )
    """
    A = matrix([[-2 * k, 1], [2 * k, 1]])
    b = vector([0, 2 * k])
    return A, b

class InteractiveMILPProblem(SageObject):
    r"""
    Construct an MILP (Mixed Integer Linear Programming) problem.

    This class supports MILP problems with "variables on the left" constraints.

    INPUT:

    - ``A`` -- a matrix of constraint coefficients

    - ``b`` -- a vector of constraint constant terms

    - ``c`` -- a vector of objective coefficients

    - ``x`` -- (default: ``"x"``) a vector of decision variables or a
      string giving the base name

    - ``constraint_type`` -- (default: ``"<="``) a string specifying constraint
      type(s): either ``"<="``, ``">="``, ``"=="``, or a list of them

    - ``variable_type`` -- (default: ``""``) a string specifying variable
      type(s): either ``">="``, ``"<="``, ``""`` (the empty string), or a
      list of them, corresponding, respectively, to non-negative,
      non-positive, and free variables

    - ``problem_type`` -- (default: ``"max"``) a string specifying the
      problem type: ``"max"``, ``"min"``, ``"-max"``, or ``"-min"``

    - ``base_ring`` -- (default: the fraction field of a common ring for all
      input coefficients) a field to which all input coefficients will be
      converted

    - ``is_primal`` -- (default: ``True``) whether this problem is primal or
      dual: each problem is of course dual to its own dual, this flag is mostly
      for internal use and affects default variable names only
      
    - ``objective_constant_term`` -- (default: 0) a constant term of the
      objective

    - ``relaxation`` -- (default: None) an :class:`LP problem <InteractiveLPProblem>`
      as the relaxation of the problem

    - ``integer_variables`` -- (default: False) either a boolean value
      indicating if all the problem variables are integer or not, or a
      set of strings giving some problem variables' names, where those
      problem variables are integer
    
    EXAMPLES:

    We will first construct the following problem directly:

    .. MATH::

        \begin{array}{l}
        \begin{array}{lcrcrcl}
         \max \mspace{-6mu}&\mspace{-6mu}  \mspace{-6mu}&\mspace{-6mu} 10 C \mspace{-6mu}&\mspace{-6mu} + \mspace{-6mu}&\mspace{-6mu} 5 B \mspace{-6mu}&\mspace{-6mu}  \mspace{-6mu}&\mspace{-6mu} \
         \mspace{-6mu}&\mspace{-6mu}  \mspace{-6mu}&\mspace{-6mu} C \mspace{-6mu}&\mspace{-6mu} + \mspace{-6mu}&\mspace{-6mu} B \mspace{-6mu}&\mspace{-6mu} \leq \mspace{-6mu}&\mspace{-6mu} 1000 \
         \mspace{-6mu}&\mspace{-6mu}  \mspace{-6mu}&\mspace{-6mu} 3 C \mspace{-6mu}&\mspace{-6mu} + \mspace{-6mu}&\mspace{-6mu} B \mspace{-6mu}&\mspace{-6mu} \leq \mspace{-6mu}&\mspace{-6mu} 1500 \
        \end{array} \\
        C, B \geq 0
        B, C \in \mathbb{Z}
        \end{array}

    ::
        sage: A = ([1, 1], [3, 1])
        sage: b = (1000, 1500)
        sage: c = (10, 5)
        sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=",
        ....:     integer_variables=True)

    Same problem, but more explicitly::

        sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], constraint_type="<=", 
        ....:     variable_type=">=", integer_variables=True)

    Even more explicitly::

        sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], problem_type="max",
        ....:     constraint_type=["<=", "<="], variable_type=[">=", ">="],
        ....:     integer_variables=True)

    Similar problem, but specifiying which decision variable is integer::

        sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], problem_type="max",
        ....:     constraint_type=["<=", "<="], variable_type=[">=", ">="],
        ....:     integer_variables={'C'})

    Using the last form you should be able to represent any MILP problem, as long
    as all like terms are collected and in constraints variables and constants
    are on different sides.

    We will construct the same problem by calling :meth:`with_relaxation` 
    in :class:`InteractiveMILPProblem`::

        sage: R = InteractiveLPProblem(A, b, c, ["C", "B"], problem_type="max",
        ....:     constraint_type=["<=", "<="], variable_type=[">=", ">="])
        sage: P = InteractiveMILPProblem.with_relaxation(R, {'C'})

    See :meth:`with_relaxation` in :class:`InteractiveMILPProblem` for more documentation. 
    """

    def __init__(self, A=None, b=None, c=None, x="x",
                 constraint_type="<=", variable_type="", 
                 problem_type="max", base_ring=None, 
                 is_primal=True, objective_constant_term=0, 
                 relaxation=None, integer_variables=False):
        r"""
        See :class:`InteractiveMILPProblem` for documentation.

        TESTS::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], constraint_type="<=", 
            ....:     variable_type=">=", integer_variables=True)
            sage: TestSuite(P).run()
        """
        if relaxation:
            if not isinstance(relaxation, InteractiveLPProblem):
                raise ValueError("relaxation should be an instance of InteractiveLPProblem")
            else:
                self._relaxation = relaxation
        else:
            self._relaxation = InteractiveLPProblem(A=A, b=b, c=c, x=x,
                                constraint_type=constraint_type, 
                                variable_type=variable_type, 
                                problem_type=problem_type, 
                                base_ring=base_ring, 
                                is_primal=is_primal, 
                                objective_constant_term=objective_constant_term)
        R = PolynomialRing(self._relaxation.base_ring(), 
                            list(self._relaxation.Abcx()[3]), order="neglex")
        if integer_variables is False:
            self._integer_variables = set([])
        elif integer_variables is True:
            self._integer_variables = set(self._relaxation.Abcx()[3])
        else:
            self._integer_variables = set([variable(R, v)
                                           for v in integer_variables])

    @classmethod
    def with_relaxation(cls, relaxation, integer_variables=False):
        r"""
        Construct a MILP problem by a relaxation and a set of integer variables.

        INPUT:

        - ``relaxation`` -- :class:`LP problem <InteractiveLPProblem>`

        - ``integer_variables`` -- (default: False) either a boolean value
          indicating if all the problem variables are integer or not, or a
          set of strings giving some problem variables' names, where those
          problem variables are integer

        OUTPUT:

        - an :class:`MILP problem <InteractiveMILPProblem>`

        EXAMPLES::

            sage: A = ([1, 1, 2], [3, 1, 7], [6, 4, 5])
            sage: b = (1000, 1500, 2000)
            sage: c = (10, 5, 1)
            sage: P = InteractiveLPProblem(A, b, c, variable_type=">=")
            sage: P1 = InteractiveMILPProblem.with_relaxation(P, True)
            sage: P1
            MILP problem (use typeset mode to see details)
            sage: P == P1.relaxation()
            True
        """
        return cls(relaxation=relaxation, integer_variables=integer_variables)

    def __eq__(self, other):
        r"""
        Check if two LP problems are equal.

        INPUT:

        - ``other`` -- anything

        OUTPUT:

        - ``True`` if ``other`` is an :class:`InteractiveLPProblem` with all details the
          same as ``self``, ``False`` otherwise.

        TESTS::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, variable_type=">=", integer_variables={"x1"})
            sage: P2 = InteractiveMILPProblem(A, b, c, variable_type=">=", integer_variables={"x1"})
            sage: P == P2
            True
            sage: P3 = InteractiveMILPProblem(A, b, c, variable_type=">=")
            sage: P == P3
            False
            sage: R = InteractiveLPProblem(A, b, c, variable_type=">=")
            sage: P4 = InteractiveMILPProblem.with_relaxation(relaxation=R, integer_variables={"x1"})
            sage: P == P4
            True
        """
        return (isinstance(other, InteractiveMILPProblem) and
                self._relaxation == other._relaxation and
                self._integer_variables == other._integer_variables)

    def _get_plot_bounding_box(self, F, b,
                              xmin=None, xmax=None, ymin=None, ymax=None):
        r"""
        Return the min and max for x and y of the bounding box for ``self``.

        INPUT:

        - ``F`` -- the feasible set of self
        - ``b`` -- the constant terms of self
        - ``xmin``, ``xmax``, ``ymin``, ``ymax`` -- bounds for the axes, if
          not given, an attempt will be made to pick reasonable values

        OUTPUT:

        - four rational numbers
        """
        if ymax is None:
            ymax = max(map(abs, b) + [v[1] for v in F.vertices()])
        if ymin is None:
            ymin = min([-ymax/4.0] + [v[1] for v in F.vertices()])
        if xmax is None:
            xmax = max([1.5*ymax] + [v[0] for v in F.vertices()])
        if xmin is None:
            xmin = min([-xmax/4.0] + [v[0] for v in F.vertices()])
        xmin, xmax, ymin, ymax = map(QQ, [xmin, xmax, ymin, ymax])
        return xmin, xmax, ymin, ymax

    def _latex_(self):
        r"""
        Return a LaTeX representation of ``self``.

        OUTPUT:

        - a string

        TESTS::

            sage: A = ([1, 1, 2], [3, 1, 7], [6, 4, 5])
            sage: b = (1000, 1500, 2000)
            sage: c = (10, 5, 1)
            sage: P = InteractiveMILPProblem(A, b, c, variable_type=">=", integer_variables={'x1'})
            sage: print(P._latex_())
            \begin{array}{l}
            \begin{array}{lcrcrcrcl}
             \max \mspace{-6mu}&\mspace{-6mu}  \mspace{-6mu}&\mspace{-6mu} 10 x_{1} \mspace{-6mu}&\mspace{-6mu} + \mspace{-6mu}&\mspace{-6mu} 5 x_{2} \mspace{-6mu}&\mspace{-6mu} + \mspace{-6mu}&\mspace{-6mu} x_{3} \mspace{-6mu}&\mspace{-6mu}  \mspace{-6mu}&\mspace{-6mu} \\
             \mspace{-6mu}&\mspace{-6mu}  \mspace{-6mu}&\mspace{-6mu} x_{1} \mspace{-6mu}&\mspace{-6mu} + \mspace{-6mu}&\mspace{-6mu} x_{2} \mspace{-6mu}&\mspace{-6mu} + \mspace{-6mu}&\mspace{-6mu} 2 x_{3} \mspace{-6mu}&\mspace{-6mu} \leq \mspace{-6mu}&\mspace{-6mu} 1000 \\
             \mspace{-6mu}&\mspace{-6mu}  \mspace{-6mu}&\mspace{-6mu} 3 x_{1} \mspace{-6mu}&\mspace{-6mu} + \mspace{-6mu}&\mspace{-6mu} x_{2} \mspace{-6mu}&\mspace{-6mu} + \mspace{-6mu}&\mspace{-6mu} 7 x_{3} \mspace{-6mu}&\mspace{-6mu} \leq \mspace{-6mu}&\mspace{-6mu} 1500 \\
             \mspace{-6mu}&\mspace{-6mu}  \mspace{-6mu}&\mspace{-6mu} 6 x_{1} \mspace{-6mu}&\mspace{-6mu} + \mspace{-6mu}&\mspace{-6mu} 4 x_{2} \mspace{-6mu}&\mspace{-6mu} + \mspace{-6mu}&\mspace{-6mu} 5 x_{3} \mspace{-6mu}&\mspace{-6mu} \leq \mspace{-6mu}&\mspace{-6mu} 2000 \\
            \end{array} \\
            x_{1}, x_{2}, x_{3} \geq 0
             \\x_{1} \in \mathbb{Z} \\x_{2}, x_{3} \in \mathbb{R}\end{array}
        """
        lines = self.relaxation()._latex_()
        integer_var = ""
        continuous_var = ""
        if self.integer_variables():
            integer_var =  r"{} \in {}".format(
                                   ", ".join(map(latex, self.integer_variables())),
                                    r"\mathbb{Z}") +  r" \\"
        if self.continuous_variables():
            continuous_var =  r"{} \in {}".format(
                                   ", ".join(map(latex, self.continuous_variables())),
                                    r"\mathbb{R}")
        return lines[:-11] + r" \\" + integer_var + continuous_var + lines[-11:]

    def _plot_constraint_or_cut(self, Ai, bi, ri, color, box, x, 
                               alpha=0.2, pad=None, ith_cut=None):
        r"""
        Return a plot of the constraint or cut of ``self``.

        INPUT:

        - ``Ai`` -- the coefficients for the constraint or cut

        - ``bi`` -- the constant for the constraint or cut

        - ``ri`` -- a string indicating the type for the constraint or cut

        - ``color`` -- a color

        - ``box`` -- a bounding box for the plot

        - ``x`` -- the decision variables of the problem

        - ``alpha`` -- (default: 0.2) determines how opaque are shadows

        - ``pad`` -- an integer

        - ``ith_cut`` -- an integer indicating the order of the cut

        OUTPUT:

        - a plot
        """
        border = box.intersection(Polyhedron(eqns=[[-bi] + list(Ai)]))
        vertices = border.vertices()
        if not vertices:
            return None
        result = Graphics()
        if not ith_cut:
            label = r"${}$".format(_latex_product(Ai, x, " ", tail=[ri, bi]))
            result += line(vertices, color=color, legend_label=label)
            if ri == "<=":
                ieqs = [[bi] + list(-Ai), [-bi+pad*Ai.norm().n()] + list(Ai)]
            elif ri == ">=":
                ieqs = [[-bi] + list(Ai), [bi+pad*Ai.norm().n()] + list(-Ai)]
            else:
                return None
            ieqs = map(lambda ieq: map(QQ, ieq), ieqs)
            halfplane = box.intersection(Polyhedron(ieqs=ieqs))
            result += halfplane.render_solid(alpha=alpha, color=color)
        else:
            label = "cut" + str(ith_cut)
            label = label + " " + r"${}$".format(
                _latex_product(Ai, x, " ", tail=[ri, bi]))
            result += line(vertices, color=color,
                           legend_label=label, thickness=1.5)
        return result

    def _repr_(self):
        r"""
        Return a string representation of ``self``.

        OUTPUT:

        - a string

        TESTS::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, variable_type=">=", integer_variables={"x1"})
            sage: print(P._repr_())
            MILP problem (use typeset mode to see details)
        """
        return "MILP problem (use typeset mode to see details)"

    def _solution(self, x):
        r"""
        Return ``x`` as a normalized solution of the relaxation of ``self``.
        
        INPUT:
                
        - ``x`` -- anything that can be interpreted as a solution of this
          problem, e.g. a vector or a list of correct length or a single
          element list with such a vector
          
        OUTPUT:
        
        - ``x`` as a vector
        
        EXAMPLES::
        
            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, variable_type=">=")
            sage: P._solution([100, 200])
            (100, 200)
            sage: P._solution([[100, 200]])
            (100, 200)
            sage: P._solution([1000])
            Traceback (most recent call last):
            ...
            TypeError: given input is not a solution for this problem
        """
        return self.relaxation()._solution(x)

    def Abcx(self):
        r"""
        Return `A`, `b`, `c`, and `x` of the relaxation of ``self`` as a tuple.

        OUTPUT:

        - a tuple

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.Abcx()
            (
            [1 1]
            [3 1], (1000, 1500), (10, 5), (C, B)
            )
        """
        return self.relaxation()._Abcx

    def add_constraint(self, coefficients, constant_term, constraint_type="<="):
        r"""
        Return a new MILP problem by adding a constraint to``self``.

        INPUT:

        - ``coefficients`` -- coefficients of the new constraint

        - ``constant_term`` -- a constant term of the new constraint

        - ``constraint_type`` -- (default: ``"<="``) a string indicating
          the constraint type of the new constraint

        OUTPUT:

        - an :class:`MILP problem <InteractiveMILPProblem>`

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c)
            sage: P1 = P.add_constraint(([2, 4]), 2000, "<=")
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
            sage: P2 = P.add_constraint(([2, 4, 6]), 2000, "<=")
            Traceback (most recent call last):
            ...
            TypeError: number of columns must be the same, not 2 and 3
            sage: P3 = P.add_constraint(([2, 4]), 2000, "<")
            Traceback (most recent call last):
            ...
            ValueError: unknown constraint type
        """
        new_relaxation = self._relaxation.add_constraint(coefficients, constant_term,
                                            constraint_type=constraint_type)
        return InteractiveMILPProblem(relaxation = new_relaxation,
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
        return set(self.relaxation().Abcx()[3])

    def base_ring(self):
        r"""
        Return the base ring of the relaxation of ``self``.

        .. NOTE::

            The base ring of MILP problems is always a field.

        OUTPUT:

        - a ring

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.base_ring()
            Rational Field

            sage: c = (10, 5.)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.base_ring()
            Real Field with 53 bits of precision
        """
        return self.relaxation().base_ring()

    def constant_terms(self):
        r"""
        Return constant terms of constraints of the relaxation of ``self``, i.e. `b`.

        OUTPUT:

        - a vector

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.constant_terms()
            (1000, 1500)
            sage: P.b()
            (1000, 1500)
        """
        return self.relaxation().constant_terms()

    def constraint_coefficients(self):
        r"""
        Return coefficients of constraints of the relaxation of ``self``, i.e. `A`.

        OUTPUT:

        - a matrix

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.constraint_coefficients()
            [1 1]
            [3 1]
            sage: P.A()
            [1 1]
            [3 1]
        """
        return self.relaxation().constraint_coefficients()

    def constraint_types(self):
        r"""
        Return a tuple listing the constraint types of all rows.

        OUTPUT:

        - a tuple of strings

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"],
            ....:     variable_type=">=", constraint_type=["<=", "=="])
            sage: P.constraint_types()
            ('<=', '==')
        """
        return self.relaxation().constraint_types()

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

    def decision_variables(self):
        r"""
        Return decision variables of the relaxation of ``self``, i.e. `x`.

        OUTPUT:

        - a vector

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.decision_variables()
            (C, B)
            sage: P.x()
            (C, B)
        """
        return self.relaxation().decision_variables()

    @cached_method
    def feasible_set(self):
        r"""
        Return the feasible set of the relaxation of ``self``.

        OUTPUT:

        - a :mod:`Polyhedron <sage.geometry.polyhedron.constructor>`

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.feasible_set()
            A 2-dimensional polyhedron in QQ^2
            defined as the convex hull of 4 vertices
        """
        return self.relaxation().feasible_set()

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

    def is_bounded(self):
        r"""
        Check if the relaxation of ``self`` is bounded.

        OUTPUT:

        - ``True`` is the relaxation of ``self`` is bounded, ``False`` otherwise

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.is_bounded()
            True
            
        Note that infeasible problems are always bounded::

            sage: b = (-1000, 1500)
            sage: P = InteractiveMILPProblem(A, b, c, variable_type=">=")
            sage: P.is_feasible()
            False
            sage: P.is_bounded()
            True
        """
        return self.relaxation().is_bounded()

    def is_feasible(self, *x):
        r"""
        Check if the relaxation of ``self`` or given solution is feasible.
        
        INPUT:
        
        - (optional) anything that can be interpreted as a valid solution for
          the relaxation of this problem, i.e. a sequence of values for all
          decision variables

        OUTPUT:

        - ``True`` is the relaxation of this problem or given solution is
          feasible, ``False`` otherwise

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, variable_type=">=")
            sage: P.is_feasible()
            True
            sage: P.is_feasible(100, 200)
            True
            sage: P.is_feasible(1000, 200)
            False
            sage: P.is_feasible([1000, 200])
            False
            sage: P.is_feasible(1000)
            Traceback (most recent call last):
            ...
            TypeError: given input is not a solution for this problem
        """
        return self.relaxation().is_feasible(*x)

    def is_negative(self):
        r"""
        Return `True` when the relaxation problem is of type ``"-max"`` or ``"-min"``.

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.is_negative()
            False
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"],
            ....:     variable_type=">=", problem_type="-min")
            sage: P.is_negative()
            True
        """
        return self.relaxation().is_negative()

    def is_optimal(self, *x):
        r"""
        Check if given solution of the relaxation is feasible.
        
        INPUT:
        
        - anything that can be interpreted as a valid solution for the relaxation
          this problem, i.e. a sequence of values for all decision variables

        OUTPUT:

        - ``True`` is the given solution is optimal, ``False`` otherwise

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (15, 5)
            sage: P = InteractiveMILPProblem(A, b, c, variable_type=">=")
            sage: P.is_optimal(100, 200)
            False
            sage: P.is_optimal(500, 0)
            True
            sage: P.is_optimal(499, 3)
            True
            sage: P.is_optimal(501, -3)
            False
        """
        return self.relaxation().is_optimal(*x)
        
    def n_constraints(self):
        r"""
        Return the number of constraints of the relaxation of ``self``, i.e. `m`.

        OUTPUT:

        - an integer

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.n_constraints()
            2
            sage: P.m()
            2
        """
        return self.relaxation().n_constraints()

    def n_variables(self):
        r"""
        Return the number of decision variables of the relaxation of ``self``, i.e. `n`.

        OUTPUT:

        - an integer

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.n_variables()
            2
            sage: P.n()
            2
        """
        return self.relaxation().n_variables()

    def objective_coefficients(self):
        r"""
        Return coefficients of the objective of ``self``, i.e. `c`.

        OUTPUT:

        - a vector

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.objective_coefficients()
            (10, 5)
            sage: P.c()
            (10, 5) 
        """
        return self.relaxation().objective_coefficients()
        
    def objective_constant_term(self):
        r"""
        Return the constant term of the objective of ``self``.

        OUTPUT:

        - a number

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.objective_constant_term()
            0
            sage: P.relaxation().optimal_value()
            6250
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"],
            ....:       variable_type=">=", objective_constant_term=-1250)
            sage: P.objective_constant_term()
            -1250
            sage: P.relaxation().optimal_value()
            5000
        """
        return self.relaxation().objective_constant_term()

    def objective_value(self, *x):
        r"""
        Return the value of the objective on the given solution of the relaxation of ``self``.
        
        INPUT:
        
        - anything that can be interpreted as a valid solution for the relaxation
          this problem, i.e. a sequence of values for all decision variables

        OUTPUT:

        - the value of the objective on the given solution taking into account
          :meth:`objective_constant_term` and :meth:`is_negative`

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, variable_type=">=")
            sage: P.objective_value(100, 200)
            2000
        """
        return self.relaxation().objective_value(*x)

    def plot(self, *args, **kwds):
        r"""
        Return a plot for solving ``self`` graphically.

        INPUT:

        - ``xmin``, ``xmax``, ``ymin``, ``ymax`` -- bounds for the axes, if
          not given, an attempt will be made to pick reasonable values

        - ``alpha`` -- (default: 0.2) determines how opaque are shadows

        OUTPUT:

        - a plot

        .. NOTE::

            This only works for problems with two decision variables.
            On the plot the black arrow indicates the direction of growth
            of the objective. The lines perpendicular to it are level
            curves of the objective. If there are optimal solutions, the
            arrow originates in one of them and the corresponding level
            curve is solid: all points of the feasible set on it are optimal
            solutions. Otherwise the arrow is placed in the center. If the
            problem is infeasible or the objective is zero, a plot of the
            feasible set only is returned.

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (100, 150)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, 
            ....:     variable_type=">=", integer_variables={'x1'})
            sage: p = P.plot()
            sage: p.show()

        In this case the plot works better with the following axes ranges::

            sage: p = P.plot(0, 1000, 0, 1500)
            sage: p.show()

        TESTS:

        We check that zero objective can be dealt with::

            sage: InteractiveMILPProblem(A, b, (0, 0),
            ....: variable_type=">=", integer_variables={'x1'}).plot()
            Graphics object consisting of 57 graphics primitives
        """
        FP = self.plot_feasible_set(*args, **kwds)
        c = self.c().n().change_ring(QQ)
        if c.is_zero():
            return FP
        if 'number_of_cuts' in kwds:
            del kwds['number_of_cuts']
        return self.plot_objective_growth_and_solution(FP, c, *args, **kwds)

    def plot_feasible_set(self, xmin=None, xmax=None, ymin=None, ymax=None,
                          alpha=0.2, number_of_cuts=0):
        r"""
        Return a plot of the feasible set of ``self``.

        INPUT:

        - ``xmin``, ``xmax``, ``ymin``, ``ymax`` -- bounds for the axes, if
          not given, an attempt will be made to pick reasonable values

        - ``alpha`` -- (default: 0.2) determines how opaque are shadows

        OUTPUT:

        - a plot

        .. NOTE::

            This only works for a problem with two decision variables. The plot
            shows boundaries of constraints with a shadow on one side for
            inequalities. If the :meth:`feasible_set` is not empty and at least
            part of it is in the given boundaries, it will be shaded gray and
            `F` will be placed in its middle.

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveLPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P1 = InteractiveMILPProblem.with_relaxation(P, True)
            sage: p = P1.plot_feasible_set()
            sage: p.show()

        In this case the plot works better with the following axes ranges::

            sage: p = P1.plot_feasible_set(0, 1000, 0, 1500)
            sage: p.show()
        """
        if self.n() != 2:
            raise ValueError("only problems with 2 variables can be plotted")
        A, b, c, x = self.Abcx()
        if self.base_ring() is not QQ:
            # Either we use QQ or crash
            A = A.n().change_ring(QQ)
            b = b.n().change_ring(QQ)
        F = self.feasible_set()
        xmin, xmax, ymin, ymax = self._get_plot_bounding_box(
            F, b, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        pad = max(xmax - xmin, ymax - ymin) / 20
        ieqs = [(xmax, -1, 0), (- xmin, 1, 0),
                (ymax, 0, -1), (- ymin, 0, 1)]
        box = Polyhedron(ieqs=ieqs)
        F = box.intersection(F)
        result = Graphics()
        colors = rainbow(self.m() + 2)
        number_of_inequalities = self.m()
        if number_of_cuts > number_of_inequalities:
            raise ValueError("number of cuts must less than number of ineqalities")
        number_of_constraints = number_of_inequalities - number_of_cuts
        list_of_number = [int(i+1) for i in range(number_of_inequalities)]

        # Plot the contraints or cuts one by one
        for i, Ai, ri, bi, color, in zip(list_of_number, A.rows(),
                                         self.constraint_types(),
                                         b, colors[:-2],):
            # Contraints are the first few number of constraints
            # inequalities of the problem
            if i <= number_of_constraints:
                plot_constraint = self._plot_constraint_or_cut(
                    Ai, bi, ri, color, box, x, alpha=alpha, pad=pad, ith_cut=None
                    )
                if plot_constraint:
                    result += plot_constraint
            # Cuts are the rest of the inequalities of the problem
            else:
                plot_cut = self._plot_constraint_or_cut(
                    Ai, bi, ri, color, box, x, alpha=alpha, pad=None,
                    ith_cut=i-number_of_constraints
                    )
                if plot_cut:
                    result += plot_cut

        # Same for variables, but no legend
        result += self.plot_variables(F, x, box, colors, pad, alpha)

        if F.vertices():
            result += F.render_solid(alpha=alpha, color="gray")
            result += text("$F$", F.center(),
                           fontsize=20, color="black", zorder=5)
        result.set_axes_range(xmin, xmax, ymin, ymax)
        result.axes_labels(["${}$".format(latex(xi)) for xi in x])
        result.legend(True)
        result.set_legend_options(fancybox=True, handlelength=1.5, loc=1,
                                  shadow=True)
        result._extra_kwds["aspect_ratio"] = 1
        result.set_aspect_ratio(1)
        return result

    def plot_lines(self, F, integer_variable):
        r"""
        Return the plot of lines (either vertical or horizontal) on an interval.

        INPUT:

        -``F`` -- the feasible set of self

        -``integer_variable`` -- a string of name of a basic integer variable
        indicating to plot vertical lines or horizontal lines

        OUTPUT:

        - a plot
        """
        b = self.b()
        xmin, xmax, ymin, ymax = self._get_plot_bounding_box(
            F, b, xmin=None, xmax=None, ymin=None, ymax=None
            )
        result = Graphics()
        for i in range(xmin, xmax+1):
            if integer_variable == "x":
                l = Polyhedron(eqns=[[-i, 1, 0]])
            else:
                l = Polyhedron(eqns=[[-i, 0, 1]])
            vertices = l.intersection(F).vertices()
            if not vertices:
                continue
            if l.intersection(F).n_vertices() == 2:
                result += line(vertices, color='blue', thickness=2)
            else:
                result += point(l.intersection(F).vertices_list(),
                                color='blue', size=22)
        return result

    def plot_objective_growth_and_solution(self, FP, c,
                                           xmin=None, xmax=None,
                                           ymin=None, ymax=None):
        r"""
        Return a plot with the growth of the objective function and the
        objective solution of the relaxation of ``self``. 

        ..Note::

            For more information, refer to the docstrings of :meth:`plot`
        in :class:`InteractiveLPProblem`.

        INPUT:

        - ``FP`` -- the plot of the feasbiel set of ``self``

        - ``c`` -- the objective value of ``self``

        - ``xmin``, ``xmax``, ``ymin``, ``ymax`` -- bounds for the axes, if
          not given, an attempt will be made to pick reasonable values

        OUTPUT:

        - a plot
        """
        b = self.b()
        xmin, xmax, ymin, ymax = self._get_plot_bounding_box(
            self.feasible_set(), b, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        start = self.relaxation().optimal_solution()
        start = vector(QQ, start.n() if start is not None
                       else [xmin + (xmax-xmin)/2, ymin + (ymax-ymin)/2])
        length = min(xmax - xmin, ymax - ymin) / 5
        end = start + (c * length / c.norm()).n().change_ring(QQ)
        result = FP + point(start, color="black", size=50, zorder=10)
        result += arrow(start, end, color="black", zorder=10)
        ieqs = [(xmax, -1, 0), (- xmin, 1, 0),
                (ymax, 0, -1), (- ymin, 0, 1)]
        box = Polyhedron(ieqs=ieqs)
        d = vector([c[1], -c[0]])
        for i in range(-10, 11):
            level = Polyhedron(vertices=[start + i*(end-start)], lines=[d])
            level = box.intersection(level)
            if level.vertices():
                if i == 0 and self.relaxation().is_bounded():
                    result += line(level.vertices(), color="black",
                                   thickness=2)
                else:
                    result += line(level.vertices(), color="black",
                                   linestyle="--")
        result.set_axes_range(xmin, xmax, ymin, ymax)
        result.axes_labels(FP.axes_labels())
        return result

    def plot_relaxation(self, *args, **kwds):
        r"""
        Return a plot for solving the relaxation of ``self`` graphically.

        INPUT:

        - ``xmin``, ``xmax``, ``ymin``, ``ymax`` -- bounds for the axes, if
          not given, an attempt will be made to pick reasonable values

        - ``alpha`` -- (default: 0.2) determines how opaque are shadows

        OUTPUT:

        - a plot

        This only works for problems with two decision variables. On the plot
        the black arrow indicates the direction of growth of the objective. The
        lines perpendicular to it are level curves of the objective. If there
        are optimal solutions, the arrow originates in one of them and the
        corresponding level curve is solid: all points of the feasible set
        on it are optimal solutions. Otherwise the arrow is placed in the
        center. If the problem is infeasible or the objective is zero, a plot
        of the feasible set only is returned.

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: p = P.plot_relaxation()
            sage: p.show()

        In this case the plot works better with the following axes ranges::

            sage: p = P.plot_relaxation(0, 1000, 0, 1500)
            sage: p.show()

        TESTS:

        We check that zero objective can be dealt with::

            sage: InteractiveMILPProblem(A, b, (0, 0), ["C", "B"],
            ....: variable_type=">=").plot_relaxation()
            Graphics object consisting of 8 graphics primitives
        """
        return self.relaxation().plot(*args, **kwds)

    def plot_variables(self, F, x, box, colors, pad, alpha):
        r"""
        Return a plot of the decision variables of ``self``

        INPUT:

        - ``F`` -- the feasible set of ``self``

        - ``x`` -- the decision variables of ``self``

        - ``colors`` -- gives a list of color

        - ``pad`` -- a number determined by xmin, xmax, ymin, ymax
          in :meth::`plot`

        - ``alpha`` -- determines how opaque are shadows

        OUTPUT:

        - a plot
        """
        if self.n() != 2:
            raise ValueError("only problems with 2 variables can be plotted")
        result = Graphics()
        integer_variables = self.integer_variables()

        # Case 1: None of the decision variables are integer
        # therefore, plot a half-plane
        # If any of the variable is an integer,
        # we will either plot integer grids or lines, but not a half-plane
        # which will be either case 2 or case 3
        if not integer_variables.intersection(set(x)):
            for ni, ri, color in zip((QQ**2).gens(), self.variable_types(),
                                     colors[-2:]):
                border = box.intersection(Polyhedron(eqns=[[0] + list(ni)]))
                if not border.vertices():
                    continue
                if ri == "<=":
                    ieqs = [[0] + list(-ni), [pad] + list(ni)]
                elif ri == ">=":
                    ieqs = [[0] + list(ni), [pad] + list(-ni)]
                else:
                    continue
                ieqs = map(lambda ieq: map(QQ, ieq), ieqs)
                halfplane = box.intersection(Polyhedron(ieqs=ieqs))
                result += halfplane.render_solid(alpha=alpha, color=color)

        # Case 2: all decision variables are integer
        # therefore, plot integer grids
        if integer_variables.intersection(set(x)) == set(x):
            feasible_dot = F.integral_points()
            result += point(feasible_dot, color='blue', alpha=1, size=22)

        # Case 3: one of the decision variables is integer, the other is not
        # therefore, plot lines
        elif x[0] in integer_variables and not x[1] in integer_variables:
            result += self.plot_lines(F, "x")
        elif x[1] in integer_variables and not x[0] in integer_variables:
            result += self.plot_lines(F, "y")
        return result

    def problem_type(self):
        r"""
        Return the problem type of the relaxation of ``self``.

        Needs to be used together with ``is_negative``.

        OUTPUT:

        - a string, one of ``"max"``, ``"min"``.

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=">=")
            sage: P.problem_type()
            'max'
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"],
            ....:     variable_type=">=", problem_type="-min")
            sage: P.problem_type()
            'min'
        """
        return self.relaxation().problem_type()

    def standard_form(self, transformation=False, **kwds):
        r"""
        Construct the MILP problem in standard form equivalent to ``self``.
        
        INPUT:
        
        - ``transformation`` -- (default: ``False``) if ``True``, a map
          converting solutions of the problem in standard form to the original
          one will be returned as well
        
        - you can pass (as keywords only) ``slack_variables``,
          ``objective_name`` to the constructor of
          :class:`InteractiveMILPProblemStandardForm`

        OUTPUT:

        - an :class:`InteractiveMILPProblemStandardForm` by itself or a tuple
          with variable transformation as the second component

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, variable_type=["<=", ""],
            ....:                            objective_constant_term=42,
            ....:                            integer_variables=True)
            sage: PSF, f = P.standard_form(True)
            sage: f
            Vector space morphism represented by the matrix:
            [-1  0]
            [ 0  1]
            [ 0 -1]
            Domain: Vector space of dimension 3 over Rational Field
            Codomain: Vector space of dimension 2 over Rational Field
            sage: PSF.relaxation().optimal_solution()
            (0, 1000, 0)
            sage: P.relaxation().optimal_solution()
            (0, 1000)
            sage: P.relaxation().is_optimal(PSF.relaxation().optimal_solution())
            Traceback (most recent call last):
            ...
            TypeError: given input is not a solution for this problem
            sage: P.relaxation().is_optimal(f(PSF.relaxation().optimal_solution()))
            True
            sage: PSF.relaxation().optimal_value()
            5042
            sage: P.relaxation().optimal_value()
            5042
            sage: P.integer_variables()
            {x1, x2}
            sage: PSF.integer_variables()
            {x1_n, x2_p, x2_n, x4, x5}


        TESTS:

        Above also works for the equivalent minimization problem::

            sage: c = (-10, -5)
            sage: P = InteractiveMILPProblem(A, b, c, variable_type=["<=", ""],
            ....:                            objective_constant_term=-42,
            ....:                            problem_type="min", 
            ....:                            integer_variables=True)
            sage: PSF, f = P.standard_form(True)
            sage: PSF.relaxation().optimal_solution()
            (0, 1000, 0)
            sage: P.relaxation().optimal_solution()
            (0, 1000)
            sage: PSF.relaxation().optimal_value()
            -5042
            sage: P.relaxation().optimal_value()
            -5042

        Integer variables are passed to standard form problem::

            sage: A = ([1, 1, 5/2], [2, 3/4, 4], [3/5, 1, 6])
            sage: b = (1000, 1500, 2000)
            sage: c = (10, 5, 3)
            sage: P = InteractiveMILPProblem(A, b, c, variable_type=[">=", "<=", ""], 
            ....:                            integer_variables=True)
            sage: PSF, f = P.standard_form(True)
            sage: P.integer_variables()
            {x1, x2, x3}
            sage: PSF.integer_variables()
            {x1, x2_n, x3_p, x3_n}

        """
        if transformation:
            (P, f) = self.relaxation().standard_form(transformation=transformation, **kwds)
        else:   
            P = self.relaxation().standard_form(transformation=transformation, **kwds)
        # assign integer variables to standard form
        I = self.integer_variables()
        x = P.Abcx()[3]
        newI = set()
        for i in tuple(I):
            for j in x:
                # variables are named as "xj_p", "xj_n", or "xj" depending on variable type
                if (str(i) == str(j)[:-2]) or (str(i) == str(j)):
                    newI.add(j)
        MIP = InteractiveMILPProblemStandardForm.with_relaxation(P, newI)
        return (MIP, f) if transformation else MIP

    def relaxation(self):
        r"""
        Return the relaxation problem of ``self``

        OUTPUT:

        - an :class:`LP problem <InteractiveLPProblem>`

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c)
            sage: R = InteractiveLPProblem(A, b, c)
            sage: P.relaxation() == R
            True
        """
        return self._relaxation

    def variable_types(self):
        r"""
        Return a tuple listing the variable types of all decision variables
        of the relaxation of ``self``.

        OUTPUT:

        - a tuple of strings

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblem(A, b, c, ["C", "B"], variable_type=[">=", ""])
            sage: P.variable_types()
            ('>=', '')
        """
        return self.relaxation().variable_types()

    # Aliases for the standard notation
    A = constraint_coefficients
    b = constant_terms
    c = objective_coefficients
    x = decision_variables
    m = n_constraints
    n = n_variables

class InteractiveMILPProblemStandardForm(InteractiveMILPProblem):
    r"""
    Construct an MILP (Mixed Integer Linear Programming) problem in standard form.

    The used standard form is:

    .. MATH::

        \begin{array}{l}
        \pm \max cx \\
        Ax \leq b \\
        x \geq 0 \\
        x might have integer components.
        \end{array}

    INPUT:

    - ``A`` -- a matrix of constraint coefficients

    - ``b`` -- a vector of constraint constant terms

    - ``c`` -- a vector of objective coefficients

    - ``x`` -- (default: ``"x"``) a vector of decision variables or a string
      the base name giving

    - ``problem_type`` -- (default: ``"max"``) a string specifying the
      problem type: either ``"max"`` or ``"-max"``

    - ``slack_variables`` -- (default: depending on :func:`style`)
      a vector of slack variables or a string giving the base name

    - ``base_ring`` -- (default: the fraction field of a common ring for all
      input coefficients) a field to which all input coefficients will be
      converted

    - ``is_primal`` -- (default: ``True``) whether this problem is primal or
      dual: each problem is of course dual to its own dual, this flag is mostly
      for internal use and affects default variable names only
      
    - ``objective_name`` -- a string or a symbolic expression for the
      objective used in dictionaries, default depending on :func:`style`

    - ``objective_constant_term`` -- (default: 0) a constant term of the
      objective

    - ``relaxation`` -- (default: None) 
      an :class:`LP problem in standard form <InteractiveLPProblemStandardForm>`
      as the relaxation of the problem

    - ``integer_variables`` -- (default: False) either a boolean value
      indicating if all the problem variables are integer or not, or a
      set of strings giving some problem variables' names, where those
      problem variables are integer

    EXAMPLES::

    We will construct the following problem directly:

        sage: A = ([1, 1], [3, 1])
        sage: b = (1000, 1500)
        sage: c = (10, 5)
        sage: P = InteractiveMILPProblemStandardForm(A, b, c, integer_variables=True)

    Unlike :class:`InteractiveMILPProblem`, this class does not allow you
    to adjust types of constraints (they are always ``"<="``) and
    variables (they are always ``">="``), and the problem type may only
    be ``"max"`` or ``"-max"``. You may give custom names to slack variables,
    but in most cases defaults should work::

        sage: P.decision_variables()
        (x1, x2)
        sage: P.slack_variables()
        (x3, x4)

    We will construct the same problem by calling :meth:`with_relaxation` 
    in :class:`InteractiveMILPProblemStandardForm`::

        sage: R = InteractiveLPProblemStandardForm(A, b, c)
        sage: P = InteractiveMILPProblem.with_relaxation(R, True)

    See :meth:`with_relaxation` in :class:`InteractiveMILPProblemStandardForm`
    for more documentation.
    """

    def __init__(self, A=None, b=None, c=None, x="x",
                 problem_type="max", slack_variables=None,
                 base_ring=None, is_primal=True, 
                 objective_name=None,
                 objective_constant_term=0, 
                 relaxation=None, integer_variables=False):
        r"""
        See :class:`InteractiveMILPProblemStandardForm` for documentation.

        TESTS::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....:     integer_variables=True)
            sage: TestSuite(P).run()
        """
        if relaxation:
            if not isinstance(relaxation, InteractiveLPProblemStandardForm):
                raise ValueError("relaxation should be an instance of InteractiveLPProblemStandardForm")
            else:
                self._relaxation = relaxation
        else:
            self._relaxation = InteractiveLPProblemStandardForm(
                                A=A, b=b, c=c, x=x, 
                                problem_type=problem_type,
                                slack_variables=slack_variables, 
                                base_ring=base_ring,
                                is_primal=is_primal,
                                objective_name=objective_name,
                                objective_constant_term=objective_constant_term)
        A = self._relaxation.Abcx()[0]
        b = self._relaxation.Abcx()[1]
        x = self._relaxation.Abcx()[3]
        R = self._relaxation._R
        m = self._relaxation.m()
        n = self._relaxation.n()
        slack_variables = self._relaxation.slack_variables()
        if integer_variables == False:
            self._integer_variables = set([])
        elif integer_variables == True:
            self._integer_variables = set(x)
        else:
            self._integer_variables = set([])
            for v in integer_variables:
                self._integer_variables.add(variable(R, v))
        # if there is no assigned integer slack variables by the user
        # use sufficient conditions to assign slack variables to be integer        
        if not self._integer_variables.intersection(set(slack_variables)): 
            if self._integer_variables.intersection(set(x)) == set(x):
                for i in range (m):
                    if b[i].is_integer() and all(coef.is_integer() for coef in A[i]):
                        self._integer_variables.add(variable(R, slack_variables[i]))
        # use sufficient conditions to assign decision variables to be integer
        # example x <= 5 where the slack variable is integer
        # then x is an integer
        if self._integer_variables.intersection(set(x)) != set(x):
            for i in range (m):
                if slack_variables[i] in self._integer_variables and b[i].is_integer():
                    for j in range (n):
                        set_Ai = copy(set(A[i]))
                        set_Ai.remove(A[i][j])
                        if A[i][j].is_integer() and set_Ai == {0}:
                            self._integer_variables.add(x[j])
                        
    @classmethod
    def with_relaxation(cls, relaxation, integer_variables=False):
        r"""
        Construct a MILP problem in standard form by a relaxation and a set of integer variables.

        INPUT:

        - ``relaxation`` -- an :class:`LP problem in standard form <InteractiveLPProblemStandardForm>`

        - ``integer_variables`` -- (default: False) either a boolean value
          indicating if all the problem variables are integer or not, or a
          set of strings giving some problem variables' names, where those
          problem variables are integer

        OUTPUT:

        - an :class:`MILP problem in standard form <InteractiveMILPProblemStandardForm>`

        EXAMPLES::

            sage: A = ([1, 1, 2], [3, 1, 7], [6, 4, 5])
            sage: b = (1000, 1500, 2000)
            sage: c = (10, 5, 1)
            sage: P = InteractiveLPProblemStandardForm(A, b, c)
            sage: P1 = InteractiveMILPProblemStandardForm.with_relaxation(P, True)
            sage: P1
            MILP problem (use typeset mode to see details)
            sage: P == P1.relaxation()
            True
        """
        return cls(relaxation=relaxation, integer_variables=integer_variables)

    def _make_Gomory_fractional_cut(self, dictionary, choose_variable, index):
        r"""
        Return the coefficients and constant for a Gomory fractional cut

        INPUT:

        - ``dictionary`` -- a :class:`dictionary <LPDictionary>`

        - ``choose_variable`` -- the basic variable for the chosen cut

        - ``index`` -- an integer indicating the choose_variable's index
          in :meth:`constant_terms`

        OUTPUT:

        - ``cut_coefficients`` -- a list of coefficients for the cut

        - ``cut_constant`` -- the constant for the cut

        EXAMPLES::

            sage: A = ([-1, 1], [8, 2])
            sage: b = (2, 17)
            sage: c = (55/10, 21/10)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables=True)
            sage: D = P.final_dictionary()
            sage: v = D.basic_variables()[0]
            sage: P._make_Gomory_fractional_cut(D, v, 0)
            ([-1/10, -4/5], -3/10)
        """
        D = dictionary
        b = D.constant_terms()
        chosen_row = D.row_coefficients(choose_variable)
        cut_coefficients = [chosen_row[i].floor() -
                            chosen_row[i] for i in range(self.n())]
        cut_constant = b[index].floor() - b[index]
        return cut_coefficients, cut_constant

    def _make_Gomory_mixed_integer_cut(self, dictionary, choose_variable, index):
        r"""
        Return the coefficients and constant a Gomory fractional cut

        INPUT:

        - ``dictionary`` -- a :class:`dictionary <LPDictionary>`

        - ``choose_variable`` -- the basic variable of the chosen cut

        - ``index`` -- an integer indicating the choose_variable's index
          in :meth:`constant_terms`

        OUTPUT:

        - ``cut_coefficients`` -- a list of coefficients for the cut

        - ``cut_constant`` -- the constant for the cut

        EXAMPLES::

            sage: A = ([-1, 1], [8, 2])
            sage: b = (2, 17)
            sage: c = (55/10, 21/10)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables=True)
            sage: D = P.final_dictionary()
            sage: v = D.basic_variables()[0]
            sage: P._make_Gomory_mixed_integer_cut(D, v, 0)
            ([-1/3, -2/7], -1)
        """
        D = dictionary
        N = D.nonbasic_variables()
        b = D.constant_terms()
        I = self.integer_variables()
        C = self.continuous_variables()
        n = self.n()

        chosen_row = D.row_coefficients(choose_variable)
        f = [chosen_row[i] - chosen_row[i].floor() for i in range(n)]
        f_0 = b[index] - b[index].floor()

        # Make dictionaries to update f and the ith row of matrix A
        # with the right orders
        # First in integer variables, then in continuous variables
        variables = list(I) + list(C)
        set_N = set(N)
        N_in_IC_order = [item for item in variables if item in set_N]
        f_dic = {item: coef for item, coef in zip(N, f)}
        new_f = [f_dic[item] for item in N_in_IC_order]
        chosen_row_dic = {item: coef for item, coef in zip(N, chosen_row)}
        new_chosen_row = [chosen_row_dic[item] for item in N_in_IC_order]

        cut_coefficients = [0] * n
        j = 0
        for item in I:
            if item in set_N:
                f_j = new_f[j]
                if f_j <= f_0:
                    cut_coefficients[j] -= f_j / f_0
                else:
                    cut_coefficients[j] -= (1 - f_j) / (1 - f_0)
                j += 1
        for item in C:
            if item in set_N:
                a_j = new_chosen_row[j]
                if a_j >= 0:
                    cut_coefficients[j] -= a_j / f_0
                else:
                    cut_coefficients[j] += a_j / (1 - f_0)
                j += 1
        cut_constant = -1

        # Update cut_coefficients in the original order
        # in self._nonbasic_variable
        cut_coef_dic = {item: coef for item, coef
                        in zip(N_in_IC_order, cut_coefficients)}
        new_cut_coefficients = [cut_coef_dic[item] for item in list(N)
                                if item in set(N_in_IC_order)]

        return new_cut_coefficients, cut_constant

    def add_a_cut(self, dictionary, integer_variables,
                  separator=None,
                  basic_variable=None, slack_variable=None):
        r"""
        Return the dictionary and the set of integer variables by adding a cut.

        INPUT:

        - ``dictionary`` -- a :class:`dictionary <LPDictionary>` or
          a :class:`revised dictionary <LPRevisedDictionary>` 

        - ``integer_variables`` -- a set of integer variables for the dictionary

        - ``separator``-- (default: None)
          a string indicating the cut generating function separator

        - ``basic_variable`` -- (default: None) a string specifying
          the basic variable that will provide the source row for the cut

        - ``slack_variable`` -- (default: None) a string giving
          the name of the slack_variable. If the argument is none,
          the new slack variable will be named as `x_n` where n is
          the next index of variable list.

        OUTPUT:

        - a :class:`dictionary <LPDictionary>` or
          a :class:`revised dictionary <LPRevisedDictionary>`

        - a set of integer variables

        EXAMPLES::

            sage: A = ([-1, 1], [8, 2])
            sage: b = (2, 17)
            sage: c = (55/10, 21/10)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables=True)
            sage: D = P.final_dictionary()
            sage: D1, I1 = P.add_a_cut(D, P.integer_variables(), 
            ....: separator="gomory_fractional")
            sage: D1.leave(D1.basic_variables()[-1])
            sage: D1.leaving_coefficients()
            (-1/10, -4/5)
            sage: D1.constant_terms()
            (33/10, 13/10, -3/10)

        The new slack variable is integer if we use Gomory fractional cut::

            sage: P.integer_variables()
            {x1, x2, x3, x4}
            sage: I1
            {x1, x2, x3, x4, x5}

        The new slack variable is continuous if we use Gomory mixed integer cut::
            
            sage: D2, I2 = P.add_a_cut(D, P.integer_variables(), 
            ....: separator="gomory_mixed_integer")
            sage: I2
            {x1, x2, x3, x4}

        Some cases of :meth:`add_a_cut` refusing making a cut by using
        Gomory fractional cut::

        1) the basic variable of the source row is not an integer::

            sage: b = (2/10, 17)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....:  integer_variables=True)
            sage: P.integer_variables()
            {x1, x2, x4}
            sage: P.add_a_cut(P.final_dictionary(), P.integer_variables(),
            ....: basic_variable="x3", separator="gomory_fractional")
            Traceback (most recent call last):
            ...
            ValueError: chosen variable should be an integer variable


        2) a non-integer variable is among the nonbasic variables 
        with non-zero coefficients on the source row::

            sage: A = ([1, 3, 5], [2, 6, 9], [6, 8, 3])
            sage: b = (12/10, 23/10, 31/10)
            sage: c = (3, 5, 7)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables= {'x1', 'x3'})
            sage: D = P.final_dictionary()
            sage: D.nonbasic_variables()
            (x6, x2, x4)
            sage: D.row_coefficients("x3")
            (-1/27, 10/27, 2/9)

        If the user chooses `x_3` to provide the source row,
        :meth:`add_a_cut` will give an error, because the non-integer
        variable `x_6` has a non-zero coefficient `1/27` on the source row::

            sage: P.add_a_cut(P.final_dictionary(), P.integer_variables(),
            ....: basic_variable='x3', separator="gomory_fractional")
            Traceback (most recent call last):
            ...
            ValueError: this is not an eligible source row

        In fact, we cannot add a Gomory fractional cut to this dictionary, because
        the non-integer variable `x_6` has non-zero coefficient on each row::

            sage: D.enter(6)
            sage: D.entering_coefficients()
            (-1/27, -1/27, 5/27)
            sage: P.add_a_cut(P.final_dictionary(), P.integer_variables(),
            ....: separator="gomory_fractional")
            Traceback (most recent call last):
            ...
            ValueError: there does not exist an eligible source row

        However, the previous restrictions are not held for Gomory
        mixed integer cuts::

            sage: D2, I2 = P.add_a_cut(P.final_dictionary(), P.integer_variables(),
            ....: separator="gomory_mixed_integer")
            sage: D2.basic_variables()
            (x3, x5, x1, x7)
        """
        # make a copy, so the original set of integer variables
        # will not be changed if we add any new integer variable
        I = copy(integer_variables)
        choose_variable, index = self.pick_eligible_source_row(
            dictionary, integer_variables,
            basic_variable=basic_variable,
            separator=separator
            )

        if separator == "gomory_mixed_integer":
            cut_coefficients, cut_constant = self._make_Gomory_mixed_integer_cut(
                dictionary, choose_variable, index)
            return dictionary.add_row(cut_coefficients, cut_constant, slack_variable), I

        elif separator == "gomory_fractional":
            cut_coefficients, cut_constant = self._make_Gomory_fractional_cut(
                dictionary, choose_variable, index)
            D = dictionary.add_row(cut_coefficients, cut_constant, slack_variable)
            # the new slack variable is integer while making a gomory fractional cut
            I.add(D.basic_variables()[-1])
            return D, I

    def add_constraint(self, coefficients, constant_term, slack_variable=None, integer_slack=False):
        r"""
        Return a new MILP problem by adding a constraint to``self``.

        INPUT:

        - ``coefficients`` -- coefficients of the new constraint

        - ``constant_term`` -- a constant term of the new constraint

        - ``slack_variable`` -- (default: depending on :func:`style`)
          a vector of the slack variable or a string giving the name

        - ``integer_slack``-- (default: False) a boolean value
          indicating if the new slack variable is integer or not.

        OUTPUT:

        - an :class:`MILP problem in standard form <InteractiveMILPProblemStandardForm>`

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: P1 = P.add_constraint(([2, 4]), 2000)
            sage: P1.Abcx()
            (
            [1 1]
            [3 1]
            [2 4], (1000, 1500, 2000), (10, 5), (x1, x2)
            )
            sage: P1.slack_variables()
            (x3, x4, x5)
            sage: P.Abcx()
            (
            [1 1]
            [3 1], (1000, 1500), (10, 5), (x1, x2)
            )
            sage: P.slack_variables()
            (x3, x4)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: P2 = P.add_constraint(([2, 4]), 2000, slack_variable='c')
            sage: P2.slack_variables()
            (x3, x4, c)
            sage: P3 = P.add_constraint(([2, 4, 6]), 2000)
            Traceback (most recent call last):
            ...
            TypeError: number of columns must be the same, not 2 and 3
        """
        new_relaxation = self.relaxation().add_constraint(coefficients, constant_term,
                                        slack_variable=slack_variable)
        integer_variables = self.integer_variables()
        if integer_slack:
            integer_variables.add(new_relaxation.slack_variables()[-1])
        return InteractiveMILPProblemStandardForm(
                    relaxation=new_relaxation,
                    integer_variables=integer_variables)

    def all_variables(self):
        r"""
        Return a set of both decision variables and slack variables of ``self``.

        OUTPUT:

        - a set of variables

        EXAMPLES::

            sage: A = ([1, 2, 1], [3, 1, 5])
            sage: b = (1000, 1500)
            sage: c = (10, 5, 7)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: P.all_variables()
            {x1, x2, x3, x4, x5}
        """
        decision_variables = self.Abcx()[3]
        slack_variables = self.slack_variables()
        all_variables = list(decision_variables) + list(slack_variables)
        return set(all_variables)

    def coordinate_ring(self):
        r"""
        Return the coordinate ring of the relaxation of ``self``.

        OUTPUT:

        - a polynomial ring over the :meth:`~InteractiveMILPProblem.base_ring` of ``self`` in
          the :meth:`auxiliary_variable`, :meth:`~InteractiveLPProblem.decision_variables`,
          and :meth:`slack_variables` with "neglex" order

        EXAMPLES::

            sage: A = ([1, 1], [3, 1], [-1, -1])
            sage: b = (1000, 1500, -400)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: P.coordinate_ring()
            Multivariate Polynomial Ring in x0, x1, x2, x3, x4, x5
            over Rational Field
            sage: P.base_ring()
            Rational Field
            sage: P.decision_variables()
            (x1, x2)
            sage: P.slack_variables()
            (x3, x4, x5)
        """
        return self.relaxation().coordinate_ring()

    def dictionary(self, *x_B):
        r"""
        Construct a dictionary for the relaxation of ``self`` with given basic variables.

        INPUT:

        - basic variables for the dictionary to be constructed

        OUTPUT:

        - a :class:`dictionary <LPDictionary>`

        .. NOTE::

            This is a synonym for ``self.revised_dictionary(x_B).dictionary()``,
            but basic variables are mandatory.

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: D = P.dictionary("x1", "x2")
            sage: D.basic_variables()
            (x1, x2)
        """
        return self.relaxation().revised_dictionary(*x_B).dictionary()

    def final_dictionary(self):
        r"""
        Return the final dictionary of the simplex method applied to 
        the relaxation of ``self``.
        
        See :meth:`run_simplex_method` for the description of possibilities.

        OUTPUT:

        - a :class:`dictionary <LPDictionary>`

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: D = P.final_dictionary()
            sage: D.is_optimal()
            True

        TESTS::

            sage: P.final_dictionary() is P.final_dictionary()
            False
        """
        return self.relaxation().final_dictionary()

    def final_revised_dictionary(self):
        r"""
        Return the final dictionary of the revised simplex method applied
        to the relaxation of ``self``.

        See :meth:`run_revised_simplex_method` for the description of
        possibilities.

        OUTPUT:

        - a :class:`revised dictionary <LPRevisedDictionary>`

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: D = P.final_revised_dictionary()
            sage: D.is_optimal()
            True

        TESTS::

            sage: P.final_revised_dictionary() is P.final_revised_dictionary()
            False 
        """
        return self.relaxation().final_revised_dictionary()

    def initial_dictionary(self):
        r"""
        Return the initial dictionary of the relaxation of ``self``.

        The initial dictionary "defines" :meth:`slack_variables` in terms
        of the :meth:`~InteractiveMILPProblem.decision_variables`, i.e.
        it has slack variables as basic ones.

        OUTPUT:

        - a :class:`dictionary <LPDictionary>`

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: D = P.initial_dictionary()
        """
        return self.relaxation().initial_dictionary()

    def inject_variables(self, scope=None, verbose=True):
        r"""
        Inject variables of ``self`` into ``scope``.

        INPUT:

        - ``scope`` -- namespace (default: global)

        - ``verbose`` -- if ``True`` (default), names of injected variables
          will be printed

        OUTPUT:

        - none

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: P.inject_variables()
            Defining x0, x1, x2, x3, x4
            sage: 3*x1 + x2
            x2 + 3*x1
        """
        return self.relaxation().inject_variables(scope=scope, verbose=verbose)

    def integer_variables(self):
        r"""
        Return the set of integer decision variables of ``self``.

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1/10, 15/10)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables={'x1'})
            sage: P.integer_variables()
            {x1}
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables=True)
            sage: P.integer_variables()
            {x1, x2}
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables=False)
            sage: P.integer_variables()
            set()

        Unlike :meth:`integer_variables` in :class:`InteractiveMILPProblem` which
        only knows about the integrality of the decision variables given by the
        user, :meth:`integer_variables` in
        :class:`InteractiveMILPProblemStandardForm` uses sufficient conditions to
        determine the integrality of decision variables, i.e. for any row, a
        decision variable `x` is an integer, if the following conditions hold:
        the slack variable of that row is set by user to be integer;
        the constant is integer;
        any decision variables except x has has a coefficient 0;
        and the coefficient of x is integer::

            sage: A1 = ([1, 1, 4], [3, 1, 5], [0, 0, 1])
            sage: b1 = (1/10, 15/10, 5)
            sage: c1 = (10, 5, 12)
            sage: P = InteractiveMILPProblemStandardForm(A1, b1, c1,
            ....: integer_variables={'x6'})
            sage: P.integer_variables()
            {x3, x6}

        Since :class:`InteractiveMILPProblemStandardForm` knows about slack
        variables, we may use the sufficient conditions to know the
        integrality of slack variables, i.e. the slack variable of a row
        is an integer if the following conditions hold:
        all decision variables are integer;
        the constant of the row is integer;
        and all coefficients of that row are integer::

            sage: A = ([1, 1], [3, 1])
            sage: b = (11, 15)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: P.integer_variables()
            set()
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables=True)
            sage: P.integer_variables()
            {x1, x2, x3, x4}
            sage: b2 = (11/10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b2, c,
            ....: integer_variables=True)
            sage: P.integer_variables()
            {x1, x2, x4}
            sage: A2 = ([1, 1], [3/10, 1])
            sage: P = InteractiveMILPProblemStandardForm(A2, b2, c,
            ....: integer_variables=True)
            sage: P.integer_variables()
            {x1, x2}

        Also, :class:`InteractiveMILPProblemStandardForm` allows the
        user to choose some slack variables to be integer, which may
        violate the sufficient conditions::

            sage: P = InteractiveMILPProblemStandardForm(A2, b2, c,
            ....: integer_variables={'x1', 'x2', 'x3'})
            sage: P.integer_variables()
            {x1, x2, x3}
        """
        return self._integer_variables

    def objective_name(self):
        r"""
        Return the objective name used in dictionaries for this problem.

        OUTPUT:
        
        - a symbolic expression

        EXAMPLES::

            sage: A = ([1, 1], [3, 1], [-1, -1])
            sage: b = (1000, 1500, -400)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: P.objective_name()
            z
            sage: sage.numerical.interactive_simplex_method.style("Vanderbei")
            'Vanderbei'
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: P.objective_name()
            zeta
            sage: sage.numerical.interactive_simplex_method.style("UAlberta")
            'UAlberta'
            sage: P = InteractiveMILPProblemStandardForm(A, b, c, objective_name="custom")
            sage: P.objective_name()
            custom
        """
        return self.relaxation().objective_name()

    def pick_eligible_source_row(self, dictionary, integer_variables,
                                 separator=None,
                                 basic_variable=None):
        r"""
        Pick an eligible source row for ``self``.

        INPUT:

        - ``dictionary`` -- a :class:`dictionary <LPDictionary>`

        - ``integer_variables`` -- the set of integer variables

        - ``separator`` -- (default: None)
          a string indicating the cut generating function separator

        - ``basic_variable`` -- (default: None) a string specifying
          the basic variable that will provide the source row for the cut

        OUTPUT:

        - ``choose_variable`` -- the basic variable for the chosen cut

        - ``index`` -- an integer indicating the choose_variable's index
          in :meth:`constant_terms`

        EXAMPLES::

            sage: A = ([-1, 1], [8, 2])
            sage: b = (2, 17)
            sage: c = (55/10, 21/10)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables=True)
            sage: D = P.final_dictionary()
            sage: D.basic_variables()
            (x2, x1)
            sage: P.integer_variables()
            {x1, x2, x3, x4}
            sage: D.constant_terms()
            (33/10, 13/10)

        None of the variables are continuous, and the constant term of `x_2` is
        not an integer. Therefore, the row for x2 is an eligible source row::

            sage: P.pick_eligible_source_row(D, P.integer_variables(),
            ....: separator="gomory_fractional")
            (x2, 0)

        See :meth:`add_a_cut` for examples of ineligible source rows
        """
        B = dictionary.basic_variables()
        list_B = list(B)
        N = dictionary.nonbasic_variables()
        b = dictionary.constant_terms()
        n = len(N)
        m = len(B)

        def eligible_source_row(
            choose_variable, bi=None,
            separator=separator
                                ):
            if separator == "gomory_fractional":
                chosen_row = dictionary.row_coefficients(choose_variable)
                for i in range(n):
                    if (N[i] not in integer_variables) and (chosen_row[i] != 0):
                        return False
            # If the choose_variable is integer and its constant is also
            # integer, then there is no need for a cut
            if (
                (not (choose_variable in integer_variables))
               or (bi is not None and bi.is_integer())):
                    return False
            return True
        integer_basic_variables = integer_variables.intersection(set(B))
        if all(b[list_B.index(variable)].is_integer()
               for variable in integer_basic_variables):
            raise ValueError(
                "the solution of the integer basic variables are all integer, " \
                "there is no way to add a cut")
        if basic_variable is not None:
            choose_variable = variable(dictionary.coordinate_ring(), basic_variable)
            if choose_variable not in integer_variables:
                raise ValueError(
                    "chosen variable should be an integer variable"
                    )
            if not eligible_source_row(choose_variable, bi=None):
                raise ValueError("this is not an eligible source row")
            index = list_B.index(choose_variable)
        else:
            fraction_list = [abs(b[i] - b[i].floor() - 0.5) for i in range(m)]
            variable_list = copy(list_B)
            while True:
                temp_index = fraction_list.index(min(fraction_list))
                # Temp index will change as long as we remove the variable of
                # the ineglible source row from the fraction list and the
                # variable lsit
                choose_variable = variable_list[temp_index]
                index = list_B.index(choose_variable)
                # Index wil not change, since we don't modify the
                # list of basic variables
                if eligible_source_row(choose_variable, b[index]):
                    break
                fraction_list.remove(min(fraction_list))
                variable_list.remove(choose_variable)
                if not fraction_list:
                    raise ValueError(
                        "there does not exist an eligible source row"
                    )
        return choose_variable, index

    def relaxation(self):
        r"""
        Return the relaxation problem of ``self``

        OUTPUT:

        - an :class:`LP problem in standard form <InteractiveLPProblemStandardForm>`

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: R = InteractiveLPProblemStandardForm(A, b, c)
            sage: P.relaxation() == R
            True
        """
        return self._relaxation

    def revised_dictionary(self, *x_B):
        r"""
        Construct a revised dictionary for the relaxation of ``self``.
    
        INPUT:

        - basic variables for the dictionary to be constructed; if not given,
          :meth:`slack_variables` will be used

        OUTPUT:

        - a :class:`revised dictionary <LPRevisedDictionary>`

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: D = P.revised_dictionary("x1", "x2")
            sage: D.basic_variables()
            (x1, x2)

        If basic variables are not given the initial dictionary is
        constructed::

            sage: P.revised_dictionary().basic_variables()
            (x3, x4)
            sage: P.initial_dictionary().basic_variables()
            (x3, x4)
        """
        return self.relaxation().revised_dictionary(*x_B)

    def run_cutting_plane_method(self, separator=None, revised=False,
                                plot=False, *args, **kwds):
        r"""
        Perform the cutting plane method to solve a MILP problem.
        Return the number of cuts needed to solve the problem and
        the final dictionary.

        INPUT:

        - ``separator`` -- (default: None) a string indicating
          the cut generating function separator

        - ``revised`` -- (default: False) a flag indicating using
          normal dictionary or revised dictionary to run the
          cutting plane method.

        - ``plot`` -- (default:False) a boolean value to decide whether
          plot the cuts or not when using revised dictionary

        - ``xmin``, ``xmax``, ``ymin``, ``ymax`` -- bounds for the axes, if
          not given, an attempt will be made to pick reasonable values

        - ``alpha`` -- (default: 0.2) determines how opaque are shadows

        OUTPUT:

        - a number which is the total number of cuts needed to solve a
          ILP problem

        - a :class:`dictionary <LPDictionary>` or
          a :class:`revised dictionary <LPRevisedDictionary>`
          depending on the ``revised``

        - :class:`~sage.misc.html.HtmlFragment` with HTML/`\LaTeX` code of
          all encountered dictionaries        

        EXAMPLES::

            sage: A = ([-1, 1], [8, 2])
            sage: b = (2, 17)
            sage: c = (55/10, 21/10)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables=True)
            sage: n, D, output = P.run_cutting_plane_method(separator="gomory_fractional", 
            ....: revised=True, plot=True, xmin=-2, xmax=6, ymin=0, ymax=12)
            sage: n
            5
            sage: type(D)
            <class 'sage.numerical.interactive_simplex_method.LPRevisedDictionary'>
            sage: output
            The original problem is:
            \begin{equation*}
            ...
            \end{equation*}
            Since $x_{2}, x_{1}, x_{4}, x_{3}$ are integer, we need to use cutting plane method to the final dictionary:\\
            \begin{equation*}
            ...
            \end{equation*}
            After adding cut 1,
            the dictionary becomes infeasible:
            \begin{equation*}
            ...
            \end{equation*}
            Run dual simplex method to solve the infeasible dictionary.\\
            The dictionary becomes:
            \begin{equation*}
            ...
            \end{equation*}
            Now we have integer variables: $x_{2}, x_{1}, x_{5}, x_{4}, x_{3}$ \\
            Since some integer variables don't have integer solutions, we need to use cutting plane method again.\\
            ...
            After adding cut 5,
            the dictionary becomes infeasible:
            \begin{equation*}
            ...
            \end{equation*}
            Run dual simplex method to solve the infeasible dictionary.\\
            The dictionary becomes:
            \begin{equation*}
            ...
            \end{equation*}
            The dictionary now is feasible and optimal.\\
            The optimal value: $\frac{59}{5}$. An optimal solution: $\left(1,\,3\right)$.

            sage: from sage.numerical.interactive_milp_problem \
            ....:     import _form_thin_long_triangle
            sage: A1, b1 = _form_thin_long_triangle(4)
            sage: c1 = (-1/27, 1/31)
            sage: P1 = InteractiveMILPProblemStandardForm(A1, b1, c1,
            ....: integer_variables=True)
            sage: n, D, output = P1.run_cutting_plane_method(
            ....: separator="gomory_fractional")
            sage: n
            9
            sage: type(D)
            <class 'sage.numerical.interactive_simplex_method.LPDictionary'>
        """
        n = 0
        output = []
        output.append("The original problem is:")
        output.append(r"\begin{equation*}")
        output.append(self._latex_())
        output.append(r"\end{equation*}")
        if revised:
            D = self.final_revised_dictionary()
        else:
            D = self.final_dictionary()

        I = self.integer_variables()
        if I == set():
            output.append("All variables are continuous, "
                          "there is no need for using cutting plane method" + r"\\")
            return n, D, HtmlFragment("\n".join(output))
        else:
            output.append("Since " + r"${}$".format(", ".join(map(latex, I))) + " are integer, "
                          "we need to use cutting plane method to the final dictionary:" + r"\\")
            output.append(D._html_())

        while True:
            D, I = self.add_a_cut(D, I, separator=separator)
            n += 1
            
            output.append("After adding cut " + str(n) + ", ")
            output.append("the dictionary becomes infeasible:")
            output.append(D._html_())
            output.append("Run dual simplex method to solve the infeasible dictionary." + r"\\")
            
            D.run_dual_simplex_method()
            output.append("The dictionary becomes:")
            output.append(D._html_())
            
            B = D.basic_variables()
            I_basic = set(B).intersection(I)
            I_indices = [tuple(B).index(v) for v in tuple(I_basic)]
            I_constant = [D.constant_terms()[i] for i in I_indices]
            if all(i.is_integer() for i in I_constant):
                break
            if n > 0:
                output.append("Now we have integer variables: "
                              r"${}$".format(", ".join(map(latex, I))) +  r" \\")
                output.append("Since some integer variables don't have integer solutions, "
                              "we need to use cutting plane method again." + r"\\")

        output.append("The dictionary now is feasible and optimal." + r"\\")
        output.append(("The optimal value: ${}$. "
                       "An optimal solution: ${}$.").format(
                       latex(- D.objective_value() if self.is_negative() else D.objective_value()), 
                       latex(D.basic_solution())))
        # we only plot the final problem when we use revised dictionary
        # since only revised dictionary knows the original problem variables
        if plot and revised:
            P = InteractiveMILPProblemStandardForm.with_relaxation(D._problem, integer_variables=I)
            result = P.plot(number_of_cuts=n, *args, **kwds)
            result.show()
        return n, D, HtmlFragment("\n".join(output))

    def run_revised_simplex_method(self):
        r"""
        Apply the revised simplex method on the relaxation of ``self`` and return all steps.

        OUTPUT:

        - :class:`~sage.misc.html.HtmlFragment` with HTML/`\LaTeX` code of
          all encountered dictionaries

        .. NOTE::

            You can access the :meth:`final_revised_dictionary`, which can be
            one of the following:

            - an optimal dictionary with the :meth:`auxiliary_variable` among
              :meth:`~LPRevisedDictionary.basic_variables` and a non-zero
              optimal value indicating
              that ``self`` is infeasible;

            - a non-optimal dictionary that has marked entering
              variable for which there is no choice of the leaving variable,
              indicating that ``self`` is unbounded;

            - an optimal dictionary.

        EXAMPLES::

            sage: A = ([1, 1], [3, 1], [-1, -1])
            sage: b = (1000, 1500, -400)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: P.run_revised_simplex_method()
            \begin{equation*}
            ...
            \end{equation*}
            Entering: $x_{1}$. Leaving: $x_{0}$.
            \begin{equation*}
            ...
            \end{equation*}
            Entering: $x_{5}$. Leaving: $x_{4}$.
            \begin{equation*}
            ...
            \end{equation*}
            Entering: $x_{2}$. Leaving: $x_{3}$.
            \begin{equation*}
            ...
            \end{equation*}
            The optimal value: $6250$. An optimal solution: $\left(250,\,750\right)$. 
        """
        return self.relaxation().run_revised_simplex_method()

    def run_simplex_method(self):
        r"""
        Apply the simplex method on the relaxation of ``self`` and return
        all steps and intermediate states.

        OUTPUT:

        - :class:`~sage.misc.html.HtmlFragment` with HTML/`\LaTeX` code of
          all encountered dictionaries

        .. NOTE::

            You can access the :meth:`final_dictionary`, which can be one
            of the following:

            - an optimal dictionary for the :meth:`auxiliary_problem` with a
              non-zero optimal value indicating that ``self`` is infeasible;

            - a non-optimal dictionary for ``self`` that has marked entering
              variable for which there is no choice of the leaving variable,
              indicating that ``self`` is unbounded;

            - an optimal dictionary for ``self``.

        EXAMPLES::

            sage: A = ([1, 1], [3, 1], [-1, -1])
            sage: b = (1000, 1500, -400)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: P.run_simplex_method()
            \begin{equation*}
            ...
            \end{equation*}
            The initial dictionary is infeasible, solving auxiliary problem.
            ...
            Entering: $x_{0}$. Leaving: $x_{5}$.
            ...
            Entering: $x_{1}$. Leaving: $x_{0}$.
            ...
            Back to the original problem.
            ...
            Entering: $x_{5}$. Leaving: $x_{4}$.
            ...
            Entering: $x_{2}$. Leaving: $x_{3}$.
            ...
            The optimal value: $6250$. An optimal solution: $\left(250,\,750\right)$. 
        """
        return self.relaxation().run_simplex_method()

    def slack_variables(self):
        r"""
        Return slack variables of ``self``.

        Slack variables are differences between the constant terms and
        left hand sides of the constraints.

        If you want to give custom names to slack variables, you have to do so
        during construction of the problem.

        OUTPUT:

        - a tuple

        EXAMPLES::

            sage: A = ([1, 1], [3, 1])
            sage: b = (1000, 1500)
            sage: c = (10, 5)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c)
            sage: P.slack_variables()
            (x3, x4)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c, ["C", "B"],
            ....:     slack_variables=["L", "F"])
            sage: P.slack_variables()
            (L, F) 
        """
        return self.relaxation().slack_variables()

# FIXME: Current code is inefficient to deal with dictionaries.
# In :meth:`run_cutting_plane_method`, one now has to check all the basic
# variables are integer or not. 
# It will be more efficient to check only the problem_variables.
# A better dictionary interface would help.