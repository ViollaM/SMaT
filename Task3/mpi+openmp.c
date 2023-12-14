//
//  seriall.c
//  Task2
//
//  Created by Маргарита Яковлева on 10.11.2023.
//

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

static double delta = 0.000001;
static int N = 40, M = 40;

struct point {
    double x;
    double y;
};
static struct point A = {-2.5, -2.5};
static struct point B = {2.5, 1.5};

struct scheme {
    double *w, *w_next, *aw;
    double *r, *ar, *b, *delta_w;
    double *Aij, *Bij;
    double *global_b, *global_w, *w_border, *r_border;
    double *w_border_global, *r_border_global;
};

static int process_id, number_of_processes;

static int domain_grid_N;
static int domain_grid_M;
static int domain_grid_border_size;
static int all_domains_grid_border_size;
static int y_domain_number;
static int domain_size;

static int grid_size;
static double h1;
static double h2;
static double eps;

int local_index(int i, int j) {
    return i * domain_grid_N + j;
}

int global_index(int i, int j) {
    return i * N + j;
}

int domain_id(int i, int j) {
    return (i / domain_grid_M) * y_domain_number + j / domain_grid_N;
}

int domain_index(int i, int j) {
    return domain_id(i, j) * domain_size + local_index(i % domain_grid_M, j % domain_grid_N);
}

int is_inside_D(struct point p) {
    return (fabs(p.x) + fabs(p.y) < 2.0) && (p.y < 1.0);
}

int is_inside_P(struct point p) {
    return p.x > A.x && p.x < B.x && p.y > A.y && p.y < B.y;
}

int is_inside_fictitious_domain(struct point p) {
    return is_inside_P(p) && ((fabs(p.x) + fabs(p.y) > 2.0) || (p.y > 1.0));
}

double F(struct point p) {
    return is_inside_D(p) ? 1.0 : 0.0;
}

double k(struct point p, double eps) {
    return is_inside_D(p) ? 1.0 : 1.0 / eps;
}

double scalar_multiplication(double *u, double *v) {
    double res = 0.0;
    register int i, j;
#pragma omp parallel for collapse(2) reduction(+:res)
    for (i = 0; i < domain_grid_M; i++) {
        for (j = 0; j < domain_grid_N; j++) {
            res += h1 * h2 * u[local_index(i, j)] * v[local_index(i, j)];
        }
    }
    return res;
}

double euclidean_norm(double *u) {
    return sqrt(scalar_multiplication(u, u));
}

// down, up
void y_borders(double x, double *y_min, double *y_max) {
    if (fabs(x) > 2) {
        *y_min = -1.0;
        *y_max = -1.0;
    }
    *y_min = fabs(x) - 2;
    *y_max = (1.0 > 2 - fabs(x)) ? 2 - fabs(x) : 1.0;
}

double aij(struct point a, struct point b, double eps) {
    double y_min, y_max;
    y_borders(a.x, &y_min, &y_max);
    double res = 0.0;
    if (is_inside_D(a) && is_inside_D(b)) {
        res = 1.0;
    } else if (is_inside_fictitious_domain(a) && is_inside_fictitious_domain(b) && ((a.y > y_max) || (b.y < y_min))) {
        res = 1.0 / eps;
    } else {
        double l_min = (b.y > y_max) ? y_max : b.y;
        double l_max = (a.y > y_min) ? a.y : y_min;
        double l = l_min - l_max;
        res = l / h2 + (1.0 - l / h2) / eps;
    }
    return res;
}

// left, right
void x_borders(double y, double *x_min, double *x_max) {
    if (y > 1 || y < -2) {
        *x_min = -1.0;
        *x_max = -1.0;
    }
    *x_min = fabs(y) - 2;
    *x_max = 2 - fabs(y);
}

double bij(struct point a, struct point b, double eps) {
    double x_min, x_max;
    x_borders(a.y, &x_min, &x_max);
    double res = 0.0;
    if (is_inside_D(a) && is_inside_D(b)) {
        res = 1.0;
    } else if (is_inside_fictitious_domain(a) && is_inside_fictitious_domain(b) && ((a.x > x_max) || (b.x < x_min))) {
        res = 1.0 / eps;
    } else {
        double l_min = (b.x > x_max) ? x_max : b.x;
        double l_max = (a.x > x_min) ? a.x : x_min;
        double l = l_min - l_max;
        res = l / h1 + (1.0 - l / h1) / eps;
    }
    return res;
}

void Aij(double *g, double eps) {
    register int i, j;
#pragma omp parallel for collapse(2)
    for (i = 1; i < M; i++) {
        for (j = 1; j < N; j++) {
            double xi = A.x + i * h1;
            double yj = A.y + j * h2;
            struct point x1 = {xi - 0.5 * h1, yj - 0.5 * h2};
            struct point x2 = {xi - 0.5 * h1, yj + 0.5 * h2};
            g[i * N + j] = aij(x1, x2, eps);
        }
    }
}

void Bij(double *g, double eps) {
    register int i, j;
#pragma omp parallel for collapse(2)
    for (i = 1; i < M; i++) {
        for (j = 1; j < N; j++) {
            double xi = A.x + i * h1;
            double yj = A.y + j * h2;
            struct point x1 = {xi - 0.5 * h1, yj - 0.5 * h2};
            struct point x2 = {xi + 0.5 * h1, yj - 0.5 * h2};
            g[i * N + j] = bij(x1, x2, eps);
        }
    }
}

void diff_operator(double *A, double *B, double *w, double *w_border, double *res) {
    register int i, j;
#pragma omp parallel for collapse(2)
    for (i = 0; i < domain_grid_M; i++) {
        for (j = 0; j < domain_grid_N; j++) {
            double a1, a2, b1, b2;
            double der_xr, der_xl, der_yr, der_yl;
            int global_i = (process_id / y_domain_number) * domain_grid_M + i, global_j = (process_id % y_domain_number) * domain_grid_N + j;
            if (global_i == 0 || global_i == M - 1 || global_j == 0 || global_j == N - 1)
                continue;

            a2 = A[global_index(global_i + 1, global_j)];
            a1 = A[global_index(global_i, global_j)];
            b2 = B[global_index(global_i, global_j + 1)];
            b1 = B[global_index(global_i, global_j)];

            der_xr = ((domain_id(global_i + 1, global_j) == process_id ? w[local_index(i + 1, j)] : w_border[(domain_id(global_i + 1, global_j)) * domain_grid_border_size + 2 * domain_grid_M + (j)]) - w[local_index(i, j)]) / h1;
            der_xl = (w[local_index(i, j)] - (domain_id(global_i - 1, global_j) == process_id ? w[local_index(i - 1, j)] : w_border[(domain_id(global_i - 1, global_j)) * domain_grid_border_size + domain_grid_M * 2 + domain_grid_N + (j)])) / h1;
            der_yr = ((domain_id(global_i, global_j + 1) == process_id ? w[local_index(i, j + 1)] : w_border[(domain_id(global_i, global_j + 1)) * domain_grid_border_size + (i)]) - w[local_index(i, j)]) / h2;
            der_yl = (w[local_index(i, j)] - (domain_id(global_i, global_j - 1) == process_id ? w[local_index(i, j - 1)] : w_border[(domain_id(global_i, global_j - 1)) * domain_grid_border_size + domain_grid_M + (i)])) / h2;


            res[local_index(i, j)] = -((a2 * der_xr - a1 * der_xl) / h1 + (b2 * der_yr - b1 * der_yl) / h2);
        }
    }
}

void linear_combination(double *g1, double c1, double *g2, double c2, double *res)
{
    register int i, j;
#pragma omp parallel for collapse(2)
    for (i = 0; i < domain_grid_M; i++) {
        for (j = 0; j < domain_grid_N; j++) {
            res[local_index(i, j)] = c1 * g1[local_index(i, j)] + c2 * g2[local_index(i, j)];
        }
    }
}

double *make_zeros_grid(int size)
{
    register int i;
    double *grid = malloc(size * sizeof(double));
#pragma omp parallel for
    for (i = 0; i < size; i++)
        grid[i] = 0.0;
    return grid;
}

double *init_grid(int size) {
    double *res = make_zeros_grid(size);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double xi = A.x + i * h1;
            double yj = A.y + j * h2;
            struct point p = {xi, yj};
            res[domain_index(i, j)] = is_inside_D(p) ? 1.0 : 0.0;
        }
    }
    return res;
}

struct scheme *init_scheme() {
    struct scheme *sch;
    sch = malloc(sizeof(*sch));

    sch->w = make_zeros_grid(domain_size);
    sch->aw = make_zeros_grid(domain_size);
    sch->w_next = make_zeros_grid(domain_size);
    sch->r_border = make_zeros_grid(domain_grid_border_size);
    sch->w_border = make_zeros_grid(domain_grid_border_size);
    sch->r_border_global = make_zeros_grid(all_domains_grid_border_size);
    sch->w_border_global = make_zeros_grid(all_domains_grid_border_size);
    sch->r = make_zeros_grid(domain_size);
    sch->ar = make_zeros_grid(domain_size);

    sch->Aij = make_zeros_grid(grid_size);
    sch->Bij = make_zeros_grid(grid_size);
    sch->b = make_zeros_grid(domain_size);

    sch->global_b = NULL;
    sch->global_w = NULL;
    if (process_id == 0) {
        sch->global_b = init_grid(grid_size);
        sch->global_w = make_zeros_grid(grid_size);
        Aij(sch->Aij, eps);
        Bij(sch->Bij, eps);
    }
    return sch;
}

void get_grid_border(double *border, const double *grid)
{
    register int i, j;
#pragma omp parallel for
    for (i = 0; i < domain_grid_M; i++)
        border[i] = grid[local_index(i, 0)];
#pragma omp parallel for
    for (i = 0; i < domain_grid_M; i++)
        border[i + domain_grid_M] = grid[local_index(i, domain_grid_N - 1)];
#pragma omp parallel for
    for (j = 0; j < domain_grid_N; j++)
        border[j + domain_grid_M + domain_grid_M] = grid[local_index(0, j)];
#pragma omp parallel for
    for (j = 0; j < domain_grid_N; j++)
        border[j + domain_grid_M + domain_grid_M + domain_grid_N] = grid[local_index(domain_grid_M - 1, j)];
}

void export_data(double *g)
{
    register int i, j;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double xi = A.x + i * h1;
            double yj = A.y + j * h2;
            fprintf(stderr, "%lf\t%lf\t%lf\n", xi, yj, g[domain_index(i, j)]);
        }
    }
}

void start() {
    double current_delta, tau, partial_delta = 0.0;
    double *tmp, local_vars[4], global_vars[4];
    struct scheme *sch = init_scheme();
    double *w_delta = make_zeros_grid(domain_size);

    MPI_Scatter(sch->global_b, domain_size, MPI_DOUBLE, sch->b, domain_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(sch->Aij, grid_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(sch->Bij, grid_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    do {
        get_grid_border(sch->w_border, sch->w);
        MPI_Allgather(sch->w_border, domain_grid_border_size, MPI_DOUBLE, sch->w_border_global, domain_grid_border_size, MPI_DOUBLE, MPI_COMM_WORLD);
        diff_operator(sch->Aij, sch->Bij, sch->w, sch->w_border_global, sch->aw);
        linear_combination(sch->aw, 1.0, sch->b, -1.0, sch->r);
        get_grid_border(sch->r_border, sch->r);
        MPI_Allgather(sch->r_border, domain_grid_border_size, MPI_DOUBLE, sch->r_border_global, domain_grid_border_size, MPI_DOUBLE, MPI_COMM_WORLD);

        diff_operator(sch->Aij, sch->Bij, sch->r, sch->r_border_global, sch->ar);
        local_vars[0] = scalar_multiplication(sch->ar, sch->r);
        local_vars[1] = pow(euclidean_norm(sch->ar), 2);
        local_vars[2] = pow(euclidean_norm(sch->r), 2);
        MPI_Allreduce(local_vars, global_vars, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tau = global_vars[0] / global_vars[1];
        linear_combination(sch->w, 1.0, sch->r, -tau, sch->w_next);
        linear_combination(sch->w, 1.0, sch->w_next, -1.0, w_delta);
        partial_delta = euclidean_norm(w_delta);
        local_vars[3] = partial_delta;
        MPI_Allreduce(local_vars, global_vars, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        current_delta = global_vars[3];

        tmp = sch->w;
        sch->w = sch->w_next;
        sch->w_next = tmp;
    } while (current_delta > delta);

    MPI_Gather(sch->w, domain_size, MPI_DOUBLE, sch->global_w, domain_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int main(int argc, const char *argv[]) {
    double start_time = 0.0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    
    double h_max;
    grid_size = N * M;
    h1 = (B.x - A.x) / (M - 1);
    h2 = (B.y - A.y) / (N - 1);
    h_max = (h1 > h2) ? h1 : h2;
    eps = pow(h_max, 2);
    int flag = 0, count = number_of_processes / 2;
    domain_grid_N = N;
    domain_grid_M = M;
    while (count > 0) {
        if (flag)
            domain_grid_N /= 2;
        else
            domain_grid_M /= 2;
        flag = !flag;
        count /= 2;
    }
    domain_grid_border_size = 2 * (domain_grid_N + domain_grid_M);
    all_domains_grid_border_size = domain_grid_border_size * number_of_processes;
    y_domain_number = N / domain_grid_N;
    domain_size = domain_grid_M * domain_grid_N;
    
    if (process_id == 0) {
        start_time = MPI_Wtime();
    }

    start();
    MPI_Barrier(MPI_COMM_WORLD);

    if (process_id == 0) {
        fprintf(stderr, "Time: %lf\n", MPI_Wtime() - start_time);
    }

    MPI_Finalize();
    return 0;
}
