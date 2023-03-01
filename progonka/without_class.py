
def create_system(n):
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0, 0] = int(random.random() * 100)
    A[0, 1] = int(random.random() * 100)
    b[0] = int(random.random() * 100)
    for i in range(1, n):
        A[i, i] = int(random.random() * 100)
        A[i, i - 1] = int(random.random() * 100)
        if i < n - 1:
            A[i, i + 1] = int(random.random() * 100)

        b[i] = int(random.random() * 100)
    return A, b


def create_np_matrix(n):
    A = np.zeros((n, n))

    A[0, 0] = int(random.random() * 100)
    A[0, 1] = int(random.random() * 100)
    for i in range(1, n):
        A[i, i] = int(random.random() * 100)
        A[i, i - 1] = int(random.random() * 100)
        if i < n - 1:
            A[i, i + 1] = int(random.random() * 100)
    return A


def thomas_solver(A, d):
    if A[0, 0] == 0:
        stderr.write('condition w[i]==0 not met')
        exit(-1)
    N = len(A)

    q = np.zeros(N)
    g = np.zeros(N)
    w = np.zeros(N)
    u = np.zeros(N)
    # forward
    q[0] = A[0, 1] / A[0, 0]
    g[0] = d[0] / A[0, 0]
    for i in range(1, N):
        w[i] = A[i, i] - A[i, i - 1] * q[i - 1]
        if w[i] == 0:
            stderr.write('condition w[i]==0 not met')
            exit(-1)
        if i != N - 1:
            q[i] = A[i, i + 1] / w[i]
        g[i] = (d[i] - A[i, i - 1] * g[i - 1]) / w[i]

    # backward
    u[N - 1] = g[N - 1]
    for i in range(N - 2, -1, -1):  # i = N-2, N-1, ... , 0
        u[i] = g[i] - q[i] * u[i + 1]

    return u

