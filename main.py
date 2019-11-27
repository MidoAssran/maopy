"""
Multi-Agent Optimization script
:author: Mido Assran
"""

import numpy as np

from arguments import get_args

from maopy.push_diging import PushDIGing
from maopy.gradient_push import GradientPush
from maopy.extra_push import ExtraPush
from maopy.asy_sonata import AsySONATA
from utils.distributed import Printer
from utils.distributed import load_peers
from utils.distributed import load_least_squares
from utils.distributed import load_softmax


def main(args):
    """ The experiment script is contained within this function. """

    # Seed algorithm
    np.random.seed(args.seed)

    # Make stdout printer
    printer = Printer(args.rank, args.size, args.comm)

    # Load peers
    if 'ring' in args.graph_file_name:
        peers = [(args.rank + 1) % args.size]
        in_degree, out_degree = len(peers), len(peers)
    else:
        peers, in_degree, out_degree = load_peers(args.graph_file_name,
                                                  args.rank,
                                                  printer)
    printer.stdout('p/o/i: %s/%s/%s' % (peers, out_degree, in_degree))

    # Load least squares data
    if 'least-squares' in args.experiment:
        objective, gradient, arg_start, arg_min = load_least_squares(
            args.data_file_name, args.rank, args.size, printer=printer)
    elif 'softmax' in args.experiment:
        objective, gradient, arg_start = load_softmax(
            args.data_file_name, args.rank, args.size, printer=printer)

    # Initialize multi-agent optimizer
    if args.alg != 'asy-sonata':
        if args.alg == 'gp':
            optimizer = GradientPush(objective=objective,
                                     sub_gradient=gradient,
                                     arg_start=arg_start,
                                     asynch=args.asynch,
                                     peers=peers,
                                     step_size=args.lr,
                                     max_itr=args.max_itr,
                                     max_time_sec=args.max_time_sec,
                                     in_degree=in_degree,
                                     tau_proc=args.tau,
                                     log=True)
        elif args.alg == 'pd':
            optimizer = PushDIGing(objective=objective,
                                   sub_gradient=gradient,
                                   arg_start=arg_start,
                                   peers=peers,
                                   step_size=args.lr,
                                   max_itr=args.max_itr,
                                   in_degree=in_degree,
                                   log=True)
        elif args.alg == 'ep':
            optimizer = ExtraPush(objective=objective,
                                  sub_gradient=gradient,
                                  arg_start=arg_start,
                                  peers=peers,
                                  step_size=args.lr,
                                  max_itr=args.max_itr,
                                  in_degree=in_degree,
                                  log=True)

        # -- log and save results
        loggers = optimizer.minimize()
        l_argmin_est = loggers['argmin_est'].history
        l_ps_w = loggers['ps_w'].history

        np.savez_compressed(args.fpath,
                            argmin_est=l_argmin_est,
                            ps_w=l_ps_w)
    else:
        optimizer = AsySONATA(objective=objective,
                              sub_gradient=gradient,
                              arg_start=arg_start,
                              peers=peers,
                              step_size=args.lr,
                              max_time_sec=args.max_time_sec,
                              in_degree=in_degree,
                              log=True)
        loggers = optimizer.minimize()
        l_argmin_est = loggers['argmin_est'].history
        np.savez_compressed(args.fpath,
                            argmin_est=l_argmin_est)

    printer.stdout('fin.')


if __name__ == '__main__':
    args = get_args()
    main(args)
