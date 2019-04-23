"""
Quadratic Program Data Generator
:author: Mido Assran
:description: Creates a quadratic program for optimization software testing.
                Based on the paper (lenard1984randomly)
              The problem data is saved into compressed numpy files.
              Problem format is:
                minimize 2-norm(Cx-d)**2
                s.t. Ax rho b
                where rho is a vector of inequality/equality constraints
"""

import sys
import numpy as np

CONFIG = {
    'features': 50,
    'data-samples': 10000,
    'condition-number': 1.,
    'ratio-final-start-objectives': 1e-5,
    'save-fpath': './datasets/qp_data_s%s.npz',
}


def make_qp(seed):
    """ The quadratic program generator script. """
    if seed is not None:
        np.random.seed(seed)

    # Load config details
    num_data_samples = CONFIG['data-samples']
    num_features = CONFIG['features']
    kai = CONFIG['condition-number'] * (seed + 1)
    gamma = CONFIG['ratio-final-start-objectives'] * (seed + 1)
    fpath = CONFIG['save-fpath'] % seed

    print('<Problem size>\n\t', num_data_samples, num_features)

    # Initialize starting guess, and final (optimal) solution
    x_0 = np.zeros(num_features)
    x_star = np.ones(num_features)

    # C = PDQ (P & Q are orthogonal, D is diagonal - sets condition number)
    p_m, _ = np.linalg.qr(np.random.randn(num_data_samples, num_data_samples))
    print('Created orthogonal matrices')
    d_m = np.zeros([num_data_samples, num_features])
    d_m[0, 0] = 1 / (kai ** 0.5)
    limitting_dimension = min(num_data_samples, num_features) - 1
    d_m[1:limitting_dimension, 1:limitting_dimension] = \
        np.diag(kai ** (np.random.uniform(-1+1e-30, 1.0,
                                          size=limitting_dimension-1) / 2.0))
    d_m[limitting_dimension, limitting_dimension] = kai ** 0.5
    c_m = p_m.dot(d_m)
    print('Computed matrix producs')
    del p_m
    del d_m
    q_m, _ = np.linalg.qr(np.random.randn(num_features, num_features))
    c_m = c_m.dot(q_m)
    del q_m
    print('Created "C" matrix')

    # Choosing 'd' is equivalent to choosing the vector of residual errors
    # d <-- C.dot(x_star) - r_star
    cx_v = c_m.dot(x_star)
    z_v = np.random.randn(num_data_samples)
    v_v = x_star - x_0
    nu = (gamma ** 2) / ((1 - (gamma ** 2)) * (np.linalg.norm(z_v) ** 2))
    vcz = v_v.dot(c_m.T).dot(z_v)
    sign = np.sign(vcz)
    beta = nu * (vcz + sign * (((vcz ** 2) + (
        (np.linalg.norm(c_m.dot(v_v)) ** 2) / nu)) ** 0.5))
    r_v = beta * z_v
    d_v = cx_v + r_v
    print('Created vector of residual errors')

    print("<Ratio of the final and start objectives>")
    print(" ", np.linalg.norm(cx_v - d_v) / np.linalg.norm(c_m.dot(x_0) - d_v))

    # Update the unconstrained optimal
    x_star = np.ones(num_features) \
        + np.linalg.inv(c_m.T.dot(c_m)).dot(c_m.T).dot(r_v)

    print("\n<f(x_star), f(x_0)>")
    print(" ",
          0.5 * np.linalg.norm(c_m.dot(x_star) - d_v)**2,
          0.5 * np.linalg.norm(c_m.dot(x_0) - d_v)**2)

    np.savez_compressed(fpath, A=c_m, x_0=x_0, x_star=x_star, b=d_v)

    print("\n<Saved variables>")
    data = np.load(fpath)
    for _, key in enumerate(data):
        print(" ", key)


def concat_qp(n):
    a_m, b_v = None, None
    for seed in range(n):
        fpath = CONFIG['save-fpath'] % seed
        data = np.load(fpath)
        if a_m is None:
            a_m = data['A']
            b_v = data['b']
        else:
            a_m = np.concatenate((a_m, data['A']), axis=0)
            b_v = np.concatenate((b_v, data['b']), axis=0)
    print(a_m.shape, b_v.shape)
    x_star = np.linalg.inv(a_m.T.dot(a_m)).dot(a_m.T.dot(b_v))
    fpath = CONFIG['save-fpath'] % 'g'
    np.savez_compressed(fpath, A=a_m, x_0=data['x_0'], x_star=x_star, b=b_v)
    print("\n<Saved variables>")
    data = np.load(fpath)
    print("\n<f(x_star), f(x_0)>")
    print(" ",
          0.5 * np.linalg.norm(a_m.dot(x_star) - b_v)**2,
          0.5 * np.linalg.norm(a_m.dot(data['x_0']) - b_v)**2)


if __name__ == '__main__':
    s, e = int(sys.argv[1]), int(sys.argv[2])
    for seed in range(s, e):
        make_qp(seed)
    concat_qp(n=e)
