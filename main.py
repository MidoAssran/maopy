"""
Multi-Agent Optimization script
:author: Mido Assran
"""

import numpy as np

from arguments import get_args

from maopy.push_sum_gossip_diging import PushDIGing
from maopy.push_sum_gossip_gradient_descent import PushSumSubgradientDescent
from maopy.push_sum_gossip_extra import ExtraPush
from utils.distributed import Printer
from utils.distributed import load_peers
from utils.distributed import load_least_squares


def main(args):
    """ The experiment script is contained within this function. """
    np.random.seed(args.seed)

    # Make stdout printer
    printer = Printer(args.rank, args.size, args.comm)

    # Load peers
    peers, in_degree, out_degree = load_peers(args.graph_file_name, args.rank,
                                              printer=printer)

    # Load least squares data
    objective, gradient, arg_start, _ = load_least_squares(args.data_file_name,
                                                           args.rank,
                                                           args.size,
                                                           printer=printer)

    # Initialize multi-agent optimizer
    if args.alg == 'gp':
        OptimizerClass = PushSumSubgradientDescent
    elif args.alg == 'pd':
        OptimizerClass = PushDIGing
    elif args.alg == 'ep':
        OptimizerClass = ExtraPush
    optimizer = OptimizerClass(
        objective=objective,
        sub_gradient=gradient,
        arg_start=arg_start,
        synch=(not args.asynch),
        peers=peers,
        step_size=args.lr,
        terminate_by_time=False,
        termination_condition=args.num_steps,
        log=True,
        out_degree=out_degree,
        in_degree=in_degree,
        constant_step_size=(not args.use_lr_decay),
        all_reduce=False)

    # Log and save results
    loggers = optimizer.minimize()
    l_argmin_est = loggers['argmin_est'].history
    l_ps_w = loggers['ps_w'].history

    # Load least squares data
    objective, _, _, arg_min = load_least_squares(args.data_file_name, 0, 1)
    true_obj = objective(arg_min)
    start_obj = objective(arg_start)
    final_obj = objective(loggers['argmin_est'].gossip_value)
    printer.stdout(
        '(min: %.4f)(final: %.4f)(start: %.4f)' % (
            true_obj, final_obj, start_obj)
    )
    np.savez_compressed(args.fpath,
                        argmin_est=l_argmin_est,
                        ps_w=l_ps_w)
    args.comm.Barrier()


if __name__ == '__main__':
    args = get_args()
    main(args)
