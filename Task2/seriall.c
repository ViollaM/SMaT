//
//  main.cpp
//  Task2
//
//  Created by Маргарита Яковлева on 10.11.2023.
//

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

static double delta = 0.000001;
static int N = 40, M = 40;

struct point {
    double x;
    double y;
};
static struct point A = {-2.0, -2.0};
static struct point B = {2.0, 1.1};

struct grid {
    double h1;
    double h2;
    double **values;
};

struct scheme {
    struct grid *w, *w_next, *aw;
    struct grid *r, *ar, *b, *delta_w;
    struct grid *Aij, *Bij;
};

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


double scalar_multiplication(struct grid *u, struct grid *v) {
    double res = 0.0;
    double h1 = u->h1, h2 = u->h2;
    for (int i = 1; i < M; i++) {
        for (int j = 1; j < N; j++) {
            res += h1 * h2 * u->values[i][j] * v->values[i][j];
        }
    }
    return res;
}

double euclidean_norm(struct grid *u) {
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

double aij(struct point a, struct point b, double h2, double eps) {
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

double bij(struct point a, struct point b, double h1, double eps) {
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

void Aij(struct grid *g, double eps) {
    double h1 = g->h1, h2 = g->h2;
    
    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= N; j++) {
            double xi = A.x + i * h1;
            double yj = A.y + j * h2;
            struct point x1 = {xi - 0.5 * h1, yj - 0.5 * h2};
            struct point x2 = {xi - 0.5 * h1, yj + 0.5 * h2};
            g->values[i][j] = aij(x1, x2, h2, eps);
        }
    }
}

void Bij(struct grid *g, double eps) {
    double h1 = g->h1, h2 = g->h2;
    
    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= N; j++) {
            double xi = A.x + i * h1;
            double yj = A.y + j * h2;
            struct point x1 = {xi - 0.5 * h1, yj - 0.5 * h2};
            struct point x2 = {xi + 0.5 * h1, yj - 0.5 * h2};
            g->values[i][j] = bij(x1, x2, h1, eps);
        }
    }
}

void diff_operator(struct grid *A, struct grid *B, struct grid *w, struct grid *res) {
    for (int i = 1; i < M; i++) {
        for (int j = 1; j < N; j++) {
            double a = -(A->values[i + 1][j] * (w->values[i + 1][j] - w->values[i][j]) - A->values[i][j] * (w->values[i][j] - w->values[i - 1][j])) / (w->h1 * w->h1);
            double b = -(B->values[i][j + 1] * (w->values[i][j + 1] - w->values[i][j]) - B->values[i][j] * (w->values[i][j] - w->values[i][j - 1])) / (w->h2 * w->h2);

            res->values[i][j] = a + b;
        }
    }
}

void linear_combination(struct grid *g1, double c1, struct grid *g2, double c2, struct grid *res)
{
    for (int i = 0; i <= M; i++) {
        for (int j = 0; j <= N; j++) {
            res->values[i][j] = c1 * g1->values[i][j] + c2 * g2->values[i][j];
        }
    }
}

struct grid *init_grid(int need_values) {
    struct grid *res;
    res = malloc(sizeof(*res));
    res->h1 = (B.x - A.x) / M;
    res->h2 = (B.y - A.y) / N;
    res->values = malloc(sizeof(double*)*(M + 1));
    for (int i = 0; i <= M; i++) {
        res->values[i] = malloc(sizeof(double)*(N + 1));
    }
    
    for (int i = 0; i <= M; i++) {
        for (int j = 0; j <= N; j++) {
            if (need_values) {
                double xi = A.x + i * res->h1;
                double yj = A.y + j * res->h2;
                struct point p = {xi, yj};
                res->values[i][j] = is_inside_D(p) ? 1.0 : 0.0;
            } else {
                res->values[i][j] = 0.0;
            }
        }
    }
    return res;
}

struct scheme *init_scheme() {
    struct scheme *sch;
    sch = malloc(sizeof(*sch));
    sch->w = init_grid(0);
    sch->w_next = init_grid(0);
    sch->aw = init_grid(0);
    sch->r = init_grid(0);
    sch->ar = init_grid(0);
    sch->b = init_grid(1);
    sch->delta_w = init_grid(0);
    sch->Aij = init_grid(0);
    sch->Bij = init_grid(0);
    
    double h_max = (sch->w->h1 > sch->w->h2) ? sch->w->h1 : sch->w->h2;
    double eps = pow(h_max, 2);
    Aij(sch->Aij, eps);
    Bij(sch->Bij, eps);
    return sch;
}

void export_data(struct grid *g)
{
    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            double xi = A.x + i * g->h1;
            double yj = A.y + j * g->h2;
            fprintf(stderr, "%lf\t%lf\t%lf\n", xi, yj, g->values[i][j]);
        }
    }
}

void free_memory_grid(struct grid *g)
{
    for (int i = 0; i <= M; i++)
        free(g->values[i]);
    free(g->values);
    free(g);
}

void free_memory(struct scheme *sch)
{
    free_memory_grid(sch->w);
    free_memory_grid(sch->aw);
    free_memory_grid(sch->w_next);
    free_memory_grid(sch->r);
    free_memory_grid(sch->ar);
    free_memory_grid(sch->b);
    free_memory_grid(sch->delta_w);
    free_memory_grid(sch->Aij);
    free_memory_grid(sch->Bij);
    free(sch);
}

void start() {
    struct scheme *sch = init_scheme();
    struct grid *tmp, *w_delta = init_grid(0);
    double tau, current_delta;

    clock_t delta_time, start_time = clock();
    do {
        diff_operator(sch->Aij, sch->Bij, sch->w, sch->aw);
        linear_combination(sch->aw, 1.0, sch->b, -1.0, sch->r);
        diff_operator(sch->Aij, sch->Bij, sch->r, sch->ar);
        tau = scalar_multiplication(sch->ar, sch->r) / scalar_multiplication(sch->ar, sch->ar);
        linear_combination(sch->w, 1.0, sch->r, -tau, sch->w_next);
        linear_combination(sch->w, 1.0, sch->w_next, -1.0, w_delta);
        current_delta = euclidean_norm(w_delta);
        tmp = sch->w;
        sch->w = sch->w_next;
        sch->w_next = tmp;
    } while (current_delta > delta);
    delta_time = clock() - start_time;
    fprintf(stderr, "time: %lf\n", (double)delta_time / (double)CLOCKS_PER_SEC);

    export_data(sch->w);
    free_memory(sch);
}

int main(int argc, const char *argv[]) {
    start();
    return 0;
}
