""" Wrapper for MPI communication variables to improve process timing. """

from numpy import empty

from mpi4py import MPI


class GossipComm(object):
    """ Wrapper class for all mpi4py communication variables. """

    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    uid = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    # Attach buffer
    # BUFF_SISE = 32064000 * (1 + MPI.BSEND_OVERHEAD)
    BUFF_SISE = 32064000 * (1 + MPI.BSEND_OVERHEAD)
    buff = empty(BUFF_SISE, dtype='b')
    MPI.Attach_buffer(buff)
