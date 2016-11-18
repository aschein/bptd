#include "csample.h"

float sample_uniform()
{
    return rand() / float(RAND_MAX);
}

bool sample_bernoulli(const double p)
{
    double u = sample_uniform();
    if (u < p) { return 1; }
    else { return 0;}
}

unsigned int sample_crt(const unsigned int m,
                        const double r)
{
    if ((r == 0) || (m == 0)) { return 0; }

    if (m == 1) { return 1; }

    unsigned int l = 0;
    for (size_t n = 1; n <= m; n++)
    {
        l = l + sample_bernoulli(r / (r + n - 1));
    }
    return l;
}