//
//  main.cpp
//  Task2
//
//  Created by Маргарита Яковлева on 10.11.2023.
//

#include <iostream>
#include <cmath>
#include <tuple>
#include <ctime>

using namespace::std;

static double delta = 0.000001;
static int N = 40, M = 40;

struct point {
    double x;
    double y;
};
static point A = point{-2.0, -2.0};
static point B = point{2.0, 1.0};

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

bool is_inside_D(point p) {
    return (abs(p.x) + abs(p.y) < 2.0) && (p.y < 1.0);
}

bool is_inside_P(point p) {
    return p.x > A.x && p.x < B.x && p.y > A.y && p.y < B.y;
}

bool is_inside_fictitious_domain(point p) {
    return is_inside_P(p) && ((abs(p.x) + abs(p.y) > 2.0) || (p.y > 1.0));
}

double F(point p) {
    return is_inside_D(p) ? 1.0 : 0.0;
}

double k(point p, double eps) {
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
tuple<double, double> y_borders(double x) {
    if (abs(x) > 2) {
        return make_tuple(-1, -1);
    }
    return make_tuple(abs(x) - 2, min(1.0, 2 - abs(x)));
}

double aij(point a, point b, double h2, double eps) {
    auto [y_min, y_max] = y_borders(a.x);
    double res = 0.0;
    if (is_inside_D(a) && is_inside_D(b)) {
        res = 1.0;
    } else if (is_inside_fictitious_domain(a) && is_inside_fictitious_domain(b) && ((a.y > y_max) || (b.y < y_min))) {
        res = 1.0 / eps;
    } else {
        double l = min(b.y, y_max) - max(a.y, y_min);
        res = l / h2 + (1.0 - l / h2) / eps;
    }
    return res;
}

// left, right
tuple<double, double> x_borders(double y) {
    if (y > 1 || y < -2) {
        return make_tuple(-1, -1);
    }
    return make_tuple(abs(y) - 2, 2 - abs(y));
}

double bij(point a, point b, double h1, double eps) {
    auto [x_min, x_max] = x_borders(a.y);
    double res = 0.0;
    if (is_inside_D(a) && is_inside_D(b)) {
        res = 1.0;
    } else if (is_inside_fictitious_domain(a) && is_inside_fictitious_domain(b) && ((a.x > x_max) || (b.x < x_min))) {
        res = 1.0 / eps;
    } else {
        double l = min(b.x, x_max) - max(a.x, x_min);
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
            g->values[i][j] = aij(point {xi - 0.5 * h1, yj - 0.5 * h2}, point {xi - 0.5 * h1, yj + 0.5 * h2}, h2, eps);
        }
    }
}

void Bij(struct grid *g, double eps) {
    double h1 = g->h1, h2 = g->h2;
    
    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= N; j++) {
            double xi = A.x + i * h1;
            double yj = A.y + j * h2;
            g->values[i][j] = bij(point {xi - 0.5 * h1, yj - 0.5 * h2}, point {xi + 0.5 * h1, yj - 0.5 * h2}, h1, eps);
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

struct grid *init_grid(bool need_values = false) {
    struct grid *res = new grid;
    res->h1 = (B.x - A.x) / M;
    res->h2 = (B.y - A.y) / N;
    res->values = new double*[M + 1];
    for (int i = 0; i <= M; i++) {
        res->values[i] = new double[N + 1];
    }
    
    for (int i = 0; i <= M; i++) {
        for (int j = 0; j <= N; j++) {
            if (need_values) {
                double xi = A.x + i * res->h1;
                double yj = A.y + j * res->h2;
                res->values[i][j] = is_inside_D(point{xi, yj}) ? 1.0 : 0.0;
            } else {
                res->values[i][j] = 0.0;
            }
        }
    }
    return res;
}

struct scheme *init_scheme() {
    struct scheme *sch = new scheme;
    sch->w = init_grid();
    sch->w_next = init_grid();
    sch->aw = init_grid();
    sch->r = init_grid();
    sch->ar = init_grid();
    sch->b = init_grid(true);
    sch->delta_w = init_grid();
    sch->Aij = init_grid();
    sch->Bij = init_grid();
    
    double h_max = max(sch->w->h1, sch->w->h2);
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

void free_memory(struct grid *g)
{
    if (g->values != nullptr) {
        for (int i = 0; i <= M; i++)
            free(g->values[i]);
        free(g->values);
        free(g);
    }
}


void free_memory(struct scheme *sch)
{
    free_memory(sch->w);
    free_memory(sch->aw);
    free_memory(sch->w_next);
    free_memory(sch->r);
    free_memory(sch->ar);
    free_memory(sch->b);
    free_memory(sch->delta_w);
    free_memory(sch->Aij);
    free_memory(sch->Bij);
    free(sch);
}

void start() {
    struct scheme *sch = init_scheme();
    struct grid *tmp = init_grid();
    double tau, current_delta;

    clock_t delta_time, start_time = clock();
    do {
        diff_operator(sch->Aij, sch->Bij, sch->w, sch->aw);
        linear_combination(sch->aw, 1.0, sch->b, -1.0, sch->r);
        diff_operator(sch->Aij, sch->Bij, sch->r, sch->ar);
        tau = scalar_multiplication(sch->ar, sch->r) / (scalar_multiplication(sch->ar, sch->ar) + 0.000001);
        linear_combination(sch->w, 1.0, sch->r, -tau, sch->w_next);
        current_delta = euclidean_norm(sch->r);
        sch->w = sch->w_next;
        sch->w_next = tmp;
    } while (current_delta > delta);
    delta_time = clock() - start_time;
    fprintf(stderr, "time: %lf\n", (double)delta_time);

    export_data(sch->w);
    free_memory(sch);
}

int main(int argc, const char *argv[]) {
    // insert code here...
//    point A = point{-1.5, 0.9};
//    point B = point{-1.5, 0.8};
//    cout << is_inside_fictitious_domain(A) << " " << is_inside_fictitious_domain(B);
//    aij(B, A, 1, 1);
//    double **s = init_grid()->values;
//    for (int i = 0; i <= M; i++) {
//        for (int j = 0; j <= N; j++) {
//            cout << s[i][j] << " ";
//        }
//        cout << endl;
//    }
    start();
    return 0;
}
