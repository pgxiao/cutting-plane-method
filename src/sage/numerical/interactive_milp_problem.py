from copy import copy
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
    def __init__(self, A=None, b=None, c=None,
                 relaxation=None, x="x",
                 constraint_type="<=", variable_type="", 
                 problem_type="max", base_ring=None, 
                 is_primal=True, objective_constant_term=0, 
                 integer_variables=False):
        if relaxation:
            if not isinstance(relaxation, InteractiveLPProblem):
                raise ValueError("relaxation should be an instance of InteractiveLPProblem")
            else:
                self._relaxation = relaxation
        else:
            self._relaxation = InteractiveLPProblem(A=A, b=b, c=c, x="x",
                                constraint_type=constraint_type, 
                                variable_type=variable_type, 
                                problem_type=problem_type, 
                                base_ring=base_ring, 
                                is_primal=is_primal, 
                                objective_constant_term=objective_constant_term)
        R = PolynomialRing(self._relaxation.base_ring(), list(self._relaxation.Abcx()[3]), order="neglex")
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
            sage: P1 = InteractiveMILPProblem.with_relaxation(P, integer_variables=True)
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
        
        See :meth:`_solution` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation()._solution(x)

    @cached_method
    def _solve(self):
        r"""
        Return an optimal solution and the optimal value of the relaxation of ``self``.

        See :meth:`_solve` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation()._solve()

    def Abcx(self):
        r"""
        Return `A`, `b`, `c`, and `x` of the relaxation of ``self`` as a tuple.

        See :meth:`Abcx` in :class:`InteractiveLPProblem` for documentation. 
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

        See :meth:`base_ring` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().base_ring()

    def constant_terms(self):
        r"""
        Return constant terms of constraints of the relaxation of ``self``, i.e. `b`.

        See :meth:`constant_terms` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().constant_terms()

    def constraint_coefficients(self):
        r"""
        Return coefficients of constraints of the relaxation of ``self``, i.e. `A`.

        See :meth:`constraint_coefficients` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().constraint_coefficients()

    def constraint_types(self):
        r"""
        Return a tuple listing the constraint types of all rows.

        See :meth:`constraint_types` in :class:`InteractiveLPProblem` for documentation. 
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

        See :meth:`decision_variables` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().decision_variables()

    @cached_method
    def feasible_set(self):
        r"""
        Return the feasible set of the relaxation of ``self``.

        See :meth:`feasible_set` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().feasible_set()

    def get_plot_bounding_box(self, F, b,
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

        See :meth:`is_bounded` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().is_bounded()

    def is_feasible(self, *x):
        r"""
        Check if the relaxation of ``self`` or given solution is feasible.
        
        See :meth:`is_feasible` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().is_feasible(*x)

    def is_negative(self):
        r"""
        Return `True` when the relaxation problem is of type ``"-max"`` or ``"-min"``.

        See :meth:`is_negative` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().is_negative()

    def is_optimal(self, *x):
        r"""
        Check if given solution of the relaxation is feasible.
        
        See :meth:`is_optimal` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().is_optimal(*x)
        
    def n_constraints(self):
        r"""
        Return the number of constraints of the relaxation of ``self``, i.e. `m`.

        See :meth:`n_constraints` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().n_constraints()

    def n_variables(self):
        r"""
        Return the number of decision variables of the relaxation of ``self``, i.e. `n`.

        See :meth:`n_variables` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().n_variables()

    def objective_coefficients(self):
        r"""
        Return coefficients of the objective of the relaxation of ``self``, i.e. `c`.

        See :meth:`objective_coefficients` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().objective_coefficients()
        
    def objective_constant_term(self):
        r"""
        Return the constant term of the objective of the relaxation of ``self``.

        See :meth:`objective_constant_term` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().objective_constant_term()

    def objective_value(self, *x):
        r"""
        Return the value of the objective on the given solution of the relaxation of ``self``.
        
        See :meth:`objective_value` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().objective_value(*x)

    def optimal_solution(self):
        r"""
        Return **an** optimal solution of the relaxation of ``self``.

        See :meth:`optimal_solution` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().optimal_solution()

    def optimal_value(self):
        r"""
        Return the optimal value for the relaxation of ``self``.

        See :meth:`optimal_value` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().optimal_value()

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
            sage: P = InteractiveMILPProblem(A, b, c, variable_type=">=", integer_variables={'x1'})
            sage: p = P.plot()
            sage: p.show()

        In this case the plot works better with the following axes ranges::

            sage: p = P.plot(0, 1000, 0, 1500)
            sage: p.show()

        TESTS:

        We check that zero objective can be dealt with::

            sage: InteractiveMILPProblem(A, b, (0, 0), variable_type=">=", integer_variables={'x1'}).plot()
            Graphics object consisting of 57 graphics primitives
        """
        FP = self.plot_feasible_set(*args, **kwds)
        c = self.c().n().change_ring(QQ)
        if c.is_zero():
            return FP
        if 'number_of_cuts' in kwds:
            del kwds['number_of_cuts']
        return self.plot_objective_growth_and_solution(FP, c, *args, **kwds)

    def plot_constraint_or_cut(self, Ai, bi, ri, color, box, x, 
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
            sage: P1 = InteractiveMILPProblem.with_relaxation(P, integer_variables=True)
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
        xmin, xmax, ymin, ymax = self.get_plot_bounding_box(
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
                plot_constraint = self.plot_constraint_or_cut(
                    Ai, bi, ri, color, box, x, alpha=alpha, pad=pad, ith_cut=None
                    )
                if plot_constraint:
                    result += plot_constraint
            # Cuts are the rest of the inequalities of the problem
            else:
                plot_cut = self.plot_constraint_or_cut(
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
        xmin, xmax, ymin, ymax = self.get_plot_bounding_box(
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
        xmin, xmax, ymin, ymax = self.get_plot_bounding_box(
            self.feasible_set(), b, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        start = self.optimal_solution()
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
                if i == 0 and self.is_bounded():
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

        See :meth:`plot` in :class:`InteractiveLPProblem` for documentation. 
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

        See :meth:`problem_type` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().problem_type()
        
    def variable_types(self):
        r"""
        Return a tuple listing the variable types of all decision variables
        of the relaxation of ``self``.

        See :meth:`variable_types` in :class:`InteractiveLPProblem` for documentation. 
        """
        return self.relaxation().variable_types()

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

    # Aliases for the standard notation
    A = constraint_coefficients
    b = constant_terms
    c = objective_coefficients
    x = decision_variables
    m = n_constraints
    n = n_variables

class InteractiveMILPProblemStandardForm(InteractiveMILPProblem):
    def __init__(self, A=None, b=None, c=None, 
                 relaxation=None,
                 x="x", problem_type="max",
                 slack_variables=None, auxiliary_variable=None,
                 base_ring=None, is_primal=True, objective_name=None,
                 objective_constant_term=0, integer_variables=False):
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
                                auxiliary_variable=auxiliary_variable,
                                base_ring=base_ring,
                                is_primal=is_primal,
                                objective_name=objective_name,
                                objective_constant_term=objective_constant_term)
        A = self._relaxation._Abcx[0]
        b = self._relaxation._Abcx[1]
        x = self._relaxation._Abcx[3]
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
            sage: P1 = InteractiveMILPProblemStandardForm.with_relaxation(P, integer_variables=True)
            sage: P1
            MILP problem (use typeset mode to see details)
            sage: P == P1.relaxation()
            True
        """
        return cls(relaxation=relaxation, integer_variables=integer_variables)

    def add_a_cut(self, dictionary, integer_variables,
                  separator=None,
                  basic_variable=None, slack_variable=None):
        r"""
        Return the dictionary by adding a cut.

        INPUT:

        - ``dictionary`` -- a :class:`dictionary <LPDictionary>`

        - ``integer_variables`` -- a set of integer variables for the dictionary

        - ``separator``-- (default: None)
          a string indicating the cut generating function separator

        - ``basic_variable`` -- (default: None) a string specifying
          the basic variable that will provide the source row for the cut

        - ``slack_variable`` -- (default: None) a string giving
          the name of the slack_variable. If the argument is none,
          the new slack variable will be the `x_n` where n is
          the next index of variable list.

        OUTPUT:

        - a :class:`dictionary <LPDictionary>`

        EXAMPLES::

            sage: A = ([-1, 1], [8, 2])
            sage: b = (2, 17)
            sage: c = (55/10, 21/10)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables=True)
            sage: D = P.final_dictionary()
            sage: D1, I = P.add_a_cut(D, P.integer_variables(), 
            ....: separator="gomory_fractional")
            sage: D1.basic_variables()
            (x2, x1, x5)
            sage: D1.leave(5)
            sage: D1.leaving_coefficients()
            (-1/10, -4/5)
            sage: D1.constant_terms()
            (33/10, 13/10, -3/10)

        :meth:`add_a_cut` refuses making a cut if the basic variable
        of the source row is not an integer::

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


        :meth:`add_a_cut` add_a_cut also refuses making a Gomory fractional
        cut if a non-integer variable is among the non-basic variables 
        with non-zero coefficients::

            sage: A = ([1, 3, 5], [2, 6, 9], [6, 8, 3])
            sage: b = (12/10, 23/10, 31/10)
            sage: c = (3, 5, 7)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables= {'x1', 'x3'})
            sage: D = P.final_dictionary()
            sage: D.nonbasic_variables()
            (x6, x2, x4)
            sage: P.integer_variables()
            {x1, x3}
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

        We cannot add a Gomory fractional cut to this dictionary, because
        the non-integer variable `x_6` has non-zero coefficient on each row::

            sage: P.add_a_cut(P.final_dictionary(), P.integer_variables(),
            ....: separator="gomory_fractional")
            Traceback (most recent call last):
            ...
            ValueError: there does not exist an eligible source row

        However, the previous condition is not necessary for Gomory
        mixed integer cuts::

            sage: D2, I = P.add_a_cut(P.final_dictionary(), P.integer_variables(),
            ....: separator="gomory_mixed_integer")
            sage: D2.basic_variables()
            (x3, x5, x1, x7)
        """
        choose_variable, index = self.pick_eligible_source_row(
            dictionary, integer_variables,
            basic_variable=basic_variable,
            separator=separator
            )

        if separator == "gomory_mixed_integer":
            cut_coefficients, cut_constant = self.make_Gomory_mixed_integer_cut(
                dictionary, choose_variable, index)
            return dictionary.add_row(cut_coefficients, cut_constant), integer_variables

        elif separator == "gomory_fractional":
            cut_coefficients, cut_constant = self.make_Gomory_fractional_cut(
                dictionary, choose_variable, index)
            D = dictionary.add_row(cut_coefficients, cut_constant)
            integer_variables.add(D.basic_variables()[-1])
            return D, integer_variables

    def add_constraint(self, coefficients, constant_term, slack_variable=None, integer_slack=False):
        r"""
        Return a new MILP problem by adding a constraint to``self``.

        INPUT:

        - ``coefficients`` -- coefficients of the new constraint

        - ``constant_term`` -- a constant term of the new constraint

        - ``slack_variable`` -- (default: depends on :func:`style`)
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

        See :meth:`coordinate_ring` in :class:`InteractiveLPProblemStandardForm` for documentation. 
        """
        return self.relaxation().coordinate_ring()

    def final_dictionary(self):
        r"""
        Return the final dictionary of the simplex method applied to 
        the relaxation of ``self``.
        
        See :meth:`final_dictionary` in :class:`InteractiveLPProblemStandardForm`
        for documentation. 
        """
        return self.relaxation().final_dictionary()

    def final_revised_dictionary(self):
        r"""
        Return the final dictionary of the revised simplex method applied
        to the relaxation of ``self``.

        See :meth:`final_revised_dictionary` in :class:`InteractiveLPProblemStandardForm`
        for documentation. 
        """
        return self.relaxation().final_revised_dictionary()

    def initial_dictionary(self):
        r"""
        Return the initial dictionary of the relaxation of ``self``.

        See :meth:`intial_dictionary` in :class:`InteractiveLPProblemStandardForm`
        for documentation. 
        """
        return self.relaxation().initial_dictionary()

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

    def make_Gomory_fractional_cut(self, dictionary, choose_variable, index):
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
            sage: P.make_Gomory_fractional_cut(D, v, 0)
            ([-1/10, -4/5], -3/10)
        """
        D = dictionary
        b = D.constant_terms()
        chosen_row = D.row_coefficients(choose_variable)
        cut_coefficients = [chosen_row[i].floor() -
                            chosen_row[i] for i in range(self.n())]
        cut_constant = b[index].floor() - b[index]
        return cut_coefficients, cut_constant

    def make_Gomory_mixed_integer_cut(self, dictionary, choose_variable, index):
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
            sage: P.make_Gomory_mixed_integer_cut(D, v, 0)
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

    def objective_name(self):
        r"""
        Return the objective name used in dictionaries for this problem.

        See :meth:`objective_name` in :class:`InteractiveLPProblemStandardForm`
        for documentation. 
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
          depends on the ``revised``

        EXAMPLES::

            sage: A = ([-1, 1], [8, 2])
            sage: b = (2, 17)
            sage: c = (55/10, 21/10)
            sage: P = InteractiveMILPProblemStandardForm(A, b, c,
            ....: integer_variables=True)
            sage: n, D = P.run_cutting_plane_method(separator="gomory_fractional", 
            ....: revised=True, plot=True, xmin=-2, xmax=6, ymin=0, ymax=12)
            sage: n
            5
            sage: type(D)
            <class 'sage.numerical.interactive_simplex_method.LPRevisedDictionary'>
            sage: from sage.numerical.interactive_milp_problem \
            ....:     import _form_thin_long_triangle
            sage: A1, b1 = _form_thin_long_triangle(4)
            sage: c1 = (-1/27, 1/31)
            sage: P1 = InteractiveMILPProblemStandardForm(A1, b1, c1,
            ....: integer_variables=True)
            sage: n, D = P1.run_cutting_plane_method(
            ....: separator="gomory_fractional")
            sage: n
            9
            sage: type(D)
            <class 'sage.numerical.interactive_simplex_method.LPDictionary'>
        """
        n = 0
        if revised:
            D = self.final_revised_dictionary()
        else:
            D = self.final_dictionary()
        I = self.integer_variables()
        # if all variables are continuous, there is no need for cutting plane method
        if I == set():
            return n, D
        while True:
            D, I = self.add_a_cut(D, I, separator=separator)
            D.run_dual_simplex_method()
            n += 1
            B = D.basic_variables()
            I_basic = set(B).intersection(I)
            I_indices = [tuple(B).index(v) for v in tuple(I_basic)]
            I_constant = [D.constant_terms()[i] for i in I_indices]
            if all(i.is_integer() for i in I_constant):
                break
        if plot and revised:
            P = InteractiveMILPProblemStandardForm.with_relaxation(D._problem, integer_variables=I)
            result = P.plot(number_of_cuts=n, *args, **kwds)
            result.show()
        return n, D

    def run_revised_simplex_method(self):
        r"""
        Apply the revised simplex method and return all steps.

        See :meth:`run_revised_simplex_method` in :class:`InteractiveLPProblemStandardForm`
        for documentation. 
        """
        return self.relaxation().run_revised_simplex_method()

    def run_simplex_method(self):
        r"""
        Apply the simplex method and return all steps and intermediate states.

        See :meth:`run_simplex_method` in :class:`InteractiveLPProblemStandardForm`
        for documentation. 
        """
        return self.relaxation().run_simplex_method()

    def slack_variables(self):
        r"""
        Return slack variables of ``self``.

        See :meth:`slack_variables` in :class:`InteractiveLPProblemStandardForm`
        for documentation. 
        """
        return self.relaxation().slack_variables()

# FIXME: Current code is inefficient to deal with dictionaries.
# In :meth:`run_cutting_plane_method`, one now has to check all the basic
# variables are integer or not. 
# It will be more efficient to check only the problem_variables.
# A better dictionary interface would help.