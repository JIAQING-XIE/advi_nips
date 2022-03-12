import math
from unittest import TestLoader
import torch

class Property():
    def __init__(self, mtx):
        self.mtx = mtx
    
    def calculate_rank(self):
        return torch.matrix_rank(self.mtx)

    def if_posdef(self):
        pass


class Cov():
    def __init__(self,):
        pass

    def create_symmetric(self, rand):
        """ rand is a random vector or matrix"""
        result = rand.matmul(rand.T)
        return result
    
    def create_posdef(self, rand):
        """ rand is a random matrix with n = m"""
        a = rand @ rand.mT + 0.01 #symmetric positive-definite
        L = torch.cholesky(a) # cholesky decompostion
        return a

    def create_triangle(self):
        pass
    
    def sim_diagonalize_matrix(self, rand):
        """ 1. judge if a matrix can do similarity diagonalization <==> exist 
        a reversible matrix P that P^-1AP is a diagonal matrix
        """
        pass


    def symmetric_to_diagonal(self, rand):
        """ rand is a real symmetric matrix"""
        """ symmetric matrix must be similar to a diagonal matrix"""
        # if rand is a symmetric matrix, then it can be transformed to a diagonal matrix
        # rand = QAQ^T or QAQ^-1 where Q is a orthogonal matrix, then we can 
        # extract A with k eigenvalues which is diagonal
        if torch.equal(ans, ans.T) == False:
            raise ValueError("Input should be a symmetric matrix")
        D, V = torch.linalg.eig(rand)
        return torch.diag(D).real
        
if __name__ == "__main__":
    cov = Cov()
    anslist = [] # all symmetrix matrix 
    # test 1: create symmetric matrix from any random matrix
    
    test1 = torch.randn(1,)
    test2 = torch.randn(2,2)
    test3 = torch.randn(2,3)

    test = [test1, test2, test3]

    for t in test:
        ans = cov.create_symmetric(t)
        if torch.equal(ans, ans.T):
            print("Symmetric matrix")
        else:
            print("Not symmetric")
            continue
        anslist.append(ans)

    # test 2: transform symmetric to diagonal (do not ensure definite)
    for ans in anslist:
        if ans.dim() == 0:
            print("yes")
            continue
        ans = cov.symmetric_to_diagonal(ans)
        print(ans)
