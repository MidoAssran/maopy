
def main():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    name = MPI.Get_processor_name()
    rank = comm.Get_rank()
    size = comm.Get_size()

    import time

    print('(%s) rank %s/%s reporting for duty' % (name, rank, size))
    tau = 1
    req = comm.Ibarrier()
    for i in range(10):
        if rank == 1:
            time.sleep(3)
        else:
            time.sleep(1)
        if req.test()[0]:
            tau = 1
            req = comm.Ibarrier()
        else:
            tau += 1
        print('(%s) %s' % (rank, tau))

    print('(%s) done' % rank)


if __name__ == '__main__':
    main()
