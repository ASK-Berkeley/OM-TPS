"""Compute the committor function of 2D MB as the solution of the Backward Kolmogorov equation using numerical quadrature
Adapted from https://github.com/muhammadhasyim/tps-torch
"""

# The below two packages are not included in the om-tps environment because it breaks torch/numpy versions.
# If you want to run this code, I recommend creating a separate conda environment
from fenics import *
from mshr import *


import matplotlib.pyplot as plt
import numpy as np
from mb_calculator import MullerBrownPotential
import os

if __name__ == "__main__":

    # Create output directory if it doesn't exist
    os.makedirs("data/committor", exist_ok=True)
    os.makedirs("data/committor/poisson", exist_ok=True)

    beta = Constant("1")  # inverse kB*T
    react_radii = 0.5
    prod_radii = 0.5
    calculator = MullerBrownPotential(device="cpu")

    react_min = np.array(calculator.initial_point)
    prod_min = np.array(calculator.final_point)

    A = np.array(calculator.A)
    alpha = np.array(calculator.alpha)
    beta_ = np.array(calculator.beta)
    gamma = np.array(calculator.gamma)
    x_ref = np.array(calculator.a)
    y_ref = np.array(calculator.b)

    rectangle = Rectangle(
        Point(calculator.Lx, calculator.Ly), Point(calculator.Hx, calculator.Hy)
    )
    react_domain = Circle(Point(react_min[0], react_min[1]), react_radii)
    prod_domain = Circle(Point(prod_min[0], prod_min[1]), prod_radii)
    domain = rectangle

    # Make subdomains
    domain.set_subdomain(1, react_domain)
    domain.set_subdomain(2, prod_domain)

    # Generate mesh
    mesh = generate_mesh(domain, 50)

    # Create boundaries
    boundary_markers = MeshFunction("size_t", mesh, 2, mesh.domains())
    boundaries_react = MeshFunction("size_t", mesh, 1, mesh.domains())
    boundaries_prod = MeshFunction("size_t", mesh, 1, mesh.domains())

    # Use the cell domains to set the boundaries
    for f in facets(mesh):
        domains = []
        for c in cells(f):
            domains.append(boundary_markers[c])
        domains = list(set(domains))
        # if len(domains) > 1:
        for i in domains:
            if i == 1:
                boundaries_react[f] = 2
            elif i == 2:
                boundaries_prod[f] = 2

    # Make function space
    V = FunctionSpace(mesh, "P", 1)

    # Define boundary conditions
    bc_react = DirichletBC(V, Constant(0), boundaries_react, 2)
    bc_prod = DirichletBC(V, Constant(1), boundaries_prod, 2)
    bcs = [bc_react, bc_prod]

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    f_0 = Expression(
        "A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))",
        degree=2,
        A_=A[0],
        a_=alpha[0],
        b_=beta_[0],
        c_=gamma[0],
        x_=x_ref[0],
        y_=y_ref[0],
    )
    f_0_exp = Expression(
        "exp(-beta*A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_)))",
        degree=2,
        A_=A[0],
        a_=alpha[0],
        b_=beta_[0],
        c_=gamma[0],
        x_=x_ref[0],
        y_=y_ref[0],
        beta=beta.values().item(),
    )
    f_0_x = Expression(
        "A_*(2*a_*(x[0]-x_)+b_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))",
        degree=2,
        A_=A[0],
        a_=alpha[0],
        b_=beta_[0],
        c_=gamma[0],
        x_=x_ref[0],
        y_=y_ref[0],
    )
    f_0_y = Expression(
        "A_*(b_*(x[0]-x_)+2*c_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))",
        degree=2,
        A_=A[0],
        a_=alpha[0],
        b_=beta_[0],
        c_=gamma[0],
        x_=x_ref[0],
        y_=y_ref[0],
    )
    f_0_grad = as_vector((f_0_x, f_0_y))

    # Repeat for f_1
    f_1 = Expression(
        "A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))",
        degree=2,
        A_=A[1],
        a_=alpha[1],
        b_=beta_[1],
        c_=gamma[1],
        x_=x_ref[1],
        y_=y_ref[1],
    )
    f_1_exp = Expression(
        "exp(-beta*A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_)))",
        degree=2,
        A_=A[1],
        a_=alpha[1],
        b_=beta_[1],
        c_=gamma[1],
        x_=x_ref[1],
        y_=y_ref[1],
        beta=beta.values().item(),
    )
    f_1_x = Expression(
        "A_*(2*a_*(x[0]-x_)+b_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))",
        degree=2,
        A_=A[1],
        a_=alpha[1],
        b_=beta_[1],
        c_=gamma[1],
        x_=x_ref[1],
        y_=y_ref[1],
    )
    f_1_y = Expression(
        "A_*(b_*(x[0]-x_)+2*c_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))",
        degree=2,
        A_=A[1],
        a_=alpha[1],
        b_=beta_[1],
        c_=gamma[1],
        x_=x_ref[1],
        y_=y_ref[1],
    )
    f_1_grad = as_vector((f_1_x, f_1_y))

    # Repeat for f_2
    f_2 = Expression(
        "A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))",
        degree=2,
        A_=A[2],
        a_=alpha[2],
        b_=beta_[2],
        c_=gamma[2],
        x_=x_ref[2],
        y_=y_ref[2],
    )
    f_2_exp = Expression(
        "exp(-beta*A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_)))",
        degree=2,
        A_=A[2],
        a_=alpha[2],
        b_=beta_[2],
        c_=gamma[2],
        x_=x_ref[2],
        y_=y_ref[2],
        beta=beta.values().item(),
    )
    f_2_x = Expression(
        "A_*(2*a_*(x[0]-x_)+b_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))",
        degree=2,
        A_=A[2],
        a_=alpha[2],
        b_=beta_[2],
        c_=gamma[2],
        x_=x_ref[2],
        y_=y_ref[2],
    )
    f_2_y = Expression(
        "A_*(b_*(x[0]-x_)+2*c_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))",
        degree=2,
        A_=A[2],
        a_=alpha[2],
        b_=beta_[2],
        c_=gamma[2],
        x_=x_ref[2],
        y_=y_ref[2],
    )
    f_2_grad = as_vector((f_2_x, f_2_y))

    # Repeat for f_3
    f_3 = Expression(
        "A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))",
        degree=2,
        A_=A[3],
        a_=alpha[3],
        b_=beta_[3],
        c_=gamma[3],
        x_=x_ref[3],
        y_=y_ref[3],
    )
    f_3_exp = Expression(
        "exp(-beta*A_*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_)))",
        degree=2,
        A_=A[3],
        a_=alpha[3],
        b_=beta_[3],
        c_=gamma[3],
        x_=x_ref[3],
        y_=y_ref[3],
        beta=beta.values().item(),
    )
    f_3_x = Expression(
        "A_*(2*a_*(x[0]-x_)+b_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))",
        degree=2,
        A_=A[3],
        a_=alpha[3],
        b_=beta_[3],
        c_=gamma[3],
        x_=x_ref[3],
        y_=y_ref[3],
    )
    f_3_y = Expression(
        "A_*(b_*(x[0]-x_)+2*c_*(x[1]-y_))*exp(a_*(x[0]-x_)*(x[0]-x_)+b_*(x[0]-x_)*(x[1]-y_)+c_*(x[1]-y_)*(x[1]-y_))",
        degree=2,
        A_=A[3],
        a_=alpha[3],
        b_=beta_[3],
        c_=gamma[3],
        x_=x_ref[3],
        y_=y_ref[3],
    )
    f_3_grad = as_vector((f_3_x, f_3_y))

    f_total = f_0 + f_1 + f_2 + f_3
    f_total_exp = f_0_exp * f_1_exp * f_2_exp * f_3_exp
    f_grad_total = f_0_grad + f_1_grad + f_2_grad + f_3_grad
    # f_total = Expression('10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)', degree=2)
    a = (dot(grad(u), grad(v)) + beta * dot(f_grad_total, grad(u)) * v) * dx
    L = Constant("0") * v * dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bcs, solver_parameters={"linear_solver": "mumps"})

    vertex_values = u.compute_vertex_values(mesh)
    coordinates = mesh.coordinates()
    np.savetxt("data/committor/vertex_values.txt", vertex_values)
    np.savetxt("data/committor/vertex_coords.txt", coordinates)

    # Evaluate u at points
    # p = Point(-0.6,0.5)

    gu = grad(u)
    Gu = project(gu, solver_type="cg")
    # Gu = dot(Gu,Gu)
    cost = assemble(dot(grad(u), grad(u)) * f_total_exp * dx)
    # print(cost)

    # Save solution to file in VTK format
    vtkfile = File("data/committor/poisson/solution.pvd")
    vtkfile << u

    # Evaluate points on structured grid
    n_struct = 100
    points_x = np.linspace(calculator.Lx, calculator.Hx, n_struct)
    points_y = np.linspace(calculator.Ly, calculator.Hy, n_struct)
    xx, yy = np.meshgrid(points_x, points_y)
    zz = np.zeros_like(xx)
    grad_2_zz = np.zeros_like(xx)
    for i in range(n_struct):
        for j in range(n_struct):
            p = Point(xx[i][j], yy[i][j])
            zz[i][j] = u(p)
            test = Gu(p)
            grad_2_zz[i][j] = test[0] * test[0] + test[1] * test[1]

    np.savetxt("data/committor/vertex_values_struct.txt", vertex_values)
    np.savetxt("data/committor/vertex_coords_struct.txt", coordinates)
    np.savetxt("data/committor/xx_structured.txt", xx)
    np.savetxt("data/committor/yy_structured.txt", yy)
    np.savetxt("data/committor/fem_committor.txt", zz)
    with open("data/committor/vertex_values_struct.txt", "w") as outf:
        for i in range(n_struct):
            for j in range(n_struct):
                outf.write("{:.6g}\n".format(zz[i][j]))

    with open("data/committor/vertex_coords_struct.txt", "w") as outf:
        for i in range(n_struct):
            for j in range(n_struct):
                outf.write("{:.6g} {:.6g}\n".format(xx[i][j], yy[i][j]))

    # Evaluate energies over grid
    grid_points = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    energies = (
        calculator.get_energy(grid_points).reshape(n_struct, n_struct).detach().numpy()
    )
    min_energy = np.min(energies)
    np.savetxt("data/committor/energies_structured.txt", grad_2_zz)
    np.savetxt(
        "data/committor/energies_factor_structured.txt",
        grad_2_zz * np.exp(-beta.values().item() * energies),
    )

    from scipy.integrate import simpson

    integral_y = simpson(
        0.5 * grad_2_zz * np.exp(-beta.values().item() * (energies - min_energy)),
        x=points_y,
    )  # Integrate over y
    integral_xy = simpson(integral_y, x=points_x)  # Integrate over y
    print(
        "BKE Loss (True Transition Rate) (Computed via Numerical Integration): ",
        integral_xy,
    )

    # Evaluate weighted gradient
    # points = np.zeros((5,2))
    # points[0,:] = np.array((-1.0,1.0))
    # points[1,:] = np.array((0.0, 0.5))
    # points[2,:] = np.array((react_min[0],react_min[1]))
    # points[3,:] = np.array((prod_min[0],prod_min[1]))
    # points[4,:] = np.array((-0.75,0.6))
    # for i in range(5):
    #     p = Point(points[i,0],points[i,1])
    #     energy_ = energy(points[i,0],points[i,1],A,a,b,c,x_,y_)
    #     grad_ = Gu(p)
    #     grad_2_ = grad_[0]**2+grad_[1]**2
    #     print(points[i,:])
    #     print(grad_2_*np.exp(-1.0*energy_))

    # Pick out values of quantity to minimize that are of high interest
    values = grad_2_zz * np.exp(-beta.values().item() * energies) > 20
    values = values.astype(int)
    np.savetxt("data/committor/values_of_interest.txt", values, fmt="%d")

    # fig, ax = plt.subplots(1,1, figsize = (7.0,2.0), dpi=600)
    h = plt.contourf(xx, yy, energies, levels=[-15 + i for i in range(16)])
    plt.colorbar()
    CS = plt.contour(
        points_x,
        points_y,
        zz,
        levels=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        cmap="Greys",
    )
    plt.colorbar()
    plt.clabel(CS, fontsize=10, inline=1)
    plt.tick_params(axis="both", which="major", labelsize=9)
    plt.tick_params(axis="both", which="minor", labelsize=9)
    plt.savefig("data/committor/committor_fem.pdf", bbox_inches="tight")
    plt.close()

    # plot energies
    h = plt.contourf(xx, yy, energies, levels=[-15 + i for i in range(16)])
    plt.colorbar()
    CS = plt.contour(xx, yy, grad_2_zz * np.exp(-1.0 * energies), cmap="Greys")
    plt.colorbar()
    plt.tick_params(axis="both", which="major", labelsize=9)
    plt.tick_params(axis="both", which="minor", labelsize=9)
    plt.savefig("data/committor/energies.pdf", bbox_inches="tight")
    plt.close()
