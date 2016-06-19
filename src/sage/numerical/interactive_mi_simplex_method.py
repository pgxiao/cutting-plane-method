from copy import copy
from sage.numerical.interactive_simplex_method import *
from sage.plot.all import Graphics, arrow, line, point, rainbow, text
from sage.rings.all import Infinity, PolynomialRing, QQ, RDF, ZZ

class InteractiveMILPProblem(SageObject):
    def __init__(self, A=None, b=None, c=None,
                 relaxation=None, x="x",
                 constraint_type="<=", variable_type="", 
                 problem_type="max", base_ring=None, 
                 is_primal=True, objective_constant_term=0, 
                 integer_variables=False):
        if relaxation:
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
            self._integer_variables = set(self._relaxation.Abcx[3])
        else:
            self._integer_variables = set([variable(R, v)
                                           for v in integer_variables])

    @classmethod
    def with_relaxation(cls, relaxation, integer_variables=False):
        return cls(relaxation=relaxation, integer_variables=integer_variables)

    def add_constraint(self, coefficients, new_b, new_constraint_type="<="):
        r"""
        Return a new MILP problem by adding a constraint to``self``.

        INPUT:

        - ``coefficients`` -- coefficients of the new constraint

        - ``new_b`` -- a constant term of the new constraint

        - ``new_constraint_type`` -- (default: ``"<="``) a string indicating
          the constraint type of the new constraint

        OUTPUT:

        - an :class:`MILP problem <InteractiveMILPProblem>`

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
        new_relaxation = self._relaxation.add_constraint(coefficients, new_b,
                                            new_constraint_type=new_constraint_type)
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

    def plot_constraint_or_cut(self, Ai, bi, ri, color, box, x, alpha,
                               pad=None, ith_cut=None):
        r"""
        Return a plot of the constraint or cut of ``self``.

        INPUT:

        - ``Ai`` -- the coefficients for the constraint or cut

        - ``bi`` -- the constant for the constraint or cut

        - ``ri`` -- a string indicating the type for the constraint or cut

        - ``color`` -- a color

        - ``box`` -- a bounding box for the plot

        - ``x`` -- the decision variables of the problem

        - ``alpha`` -- determines how opaque are shadows

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
            for ni, ri, color in zip((QQ**2).gens(), self._variable_types,
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

class InteractiveMILPProblemStandardForm(SageObject):
    def __init__(self, A=None, b=None, c=None, 
                 relaxation=None,
                 x="x", problem_type="max",
                 slack_variables=None, auxiliary_variable=None,
                 base_ring=None, is_primal=True, objective_name=None,
                 objective_constant_term=0, integer_variables=False):
        if relaxation:
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
        return cls(relaxation=relaxation, integer_variables=integer_variables)

    def add_constraint(self, coefficients, new_b, new_slack_variable=None, integer_slack=False):
        r"""
        Return a new MILP problem by adding a constraint to``self``.

        INPUT:

        - ``coefficients`` -- coefficients of the new constraint

        - ``new_b`` -- a constant term of the new constraint

        - ``new_slack_variable`` -- (default: depends on :func:`style`)
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
            sage: P2 = P.add_constraint(([2, 4]), 2000, new_slack_variable='c')
            sage: P2.slack_variables()
            (x3, x4, c)
            sage: P3 = P.add_constraint(([2, 4, 6]), 2000)
            Traceback (most recent call last):
            ...
            ValueError: A and coefficients have incompatible dimensions
        """
        new_relaxation = add_constraint(coefficients, new_b,
                                        new_slack_variable=new_slack_variable)
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