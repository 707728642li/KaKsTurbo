#ifndef KAKS_Q_SNAPSHOT_HOOK_H
#define KAKS_Q_SNAPSHOT_HOOK_H
/* Read-only diagnostics in the independently staged original GY94.cpp. */
void kaks_q_snapshot_record_q(const double *q, const double *pi, double mr,
                             const double *rates, double omega, int n);
void kaks_q_snapshot_record_s(const double *s, int n, const double *sqrt_pi);
#endif
