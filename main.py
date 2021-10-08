import numpy as np
from numpy.random import rand
from numpy.linalg import inv, norm

rank = 4
I = 16

factor_matrix = rand(I, rank) * 2 - 1

# Symmetric gram matrix
Phi = np.dot(factor_matrix.T, factor_matrix)
Psi = rand(I, rank) 
A = rand(I, rank) * 2 - 1

# Dual variable U
U = np.zeros((I, rank))
epsilon = 1e-3

"""
A_(n): Factor matrix for certain mode n
Phi_(n): Gram matrix for aTa for all other modes
Psi_(n): MTTKRP output 
"""

def admm(Phi, Psi, A, U):
    rho = np.trace(Phi) / rank
    Phi_inv = inv(Phi + rho * np.eye(rank))

    print(Phi)
    print(Phi_inv)

    it = 0
    prev_A = np.zeros(A.shape)

    print("Before constraint")
    print(A)

    while True:
        # Update A_t^(n)
        A_hat = np.dot(Psi + rho * (A + U), Phi_inv)

        # non negativity constraint
        # A = ((A_hat - U) >= 0.0) * (A_hat - U)
        # lasso reguliarizatoin
        A = ((1 - (1 / (rho * (A_hat - U)))) >= 0.0) * (A_hat - U)
        U = U - A_hat + A

        # Check convergence
        # Relative primal residual
        res_r = norm(A - A_hat) / norm(A)
        # Relative dual residual
        res_s = norm(A - prev_A) / (norm(U) + 1e-6)
        
        prev_A = A
        # print("res_r:", res_r)
        print("res_s:", res_s)
        it+=1
        if res_r < epsilon and res_s < epsilon:
            break

    print("Num of iterations: {}".format(it))
    print(A)


admm(Phi, Psi, A, U)
"""
if __name__ == "__main__":
    admm()
"""
