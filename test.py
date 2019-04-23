
def main():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    name = MPI.Get_processor_name()
    rank = comm.Get_rank()
    size = comm.Get_size()

    import time
    time.sleep(5)

    print('(%s) rank %s/%s reporting for duty' % (name, rank, size))
    if rank == 0:
            data = {'a': 7, 'b': 3.14}
            comm.send(data, dest=1, tag=11)
    elif rank == 1:
            data = comm.recv(source=0, tag=11)
    print(rank, data)


if __name__ == '__main__':
    main()
