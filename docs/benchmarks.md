# Benchmarks

GES and XGES on their own, and how they compare with other implementations, are in [causalbase](https://github.com/ptorrijos99/causalbase)'s benchmarks.

## What federating costs

Mean SHD against the true network over three samples. The first column learns from all the data in one place; the others split the same rows across that many sites.

| Network | Variables | Centralised | 5 sites | 10 sites | 20 sites |
|---|---:|---:|---:|---:|---:|
| cancer | 5 | 0.7 | 1.0 | 2.0 | 3.0 |
| asia | 8 | 2.0 | 2.3 | 2.0 | 4.3 |
| child | 20 | 2.0 | 7.0 | 10.0 | 10.0 |
| insurance | 27 | 21.7 | 26.0 | 28.0 | 29.7 |
| alarm | 37 | 10.0 | 5.0 | 5.0 | 6.7 |
| hepar2 | 70 | 62.7 | 76.7 | 96.7 | 109.7 |

Each site sees less data the more sites there are, and on most networks the error grows with the number of sites; `hepar2` goes from 63 learning centrally to 110 across twenty sites. `alarm` is the exception: the fused structure is closer to the truth than the one learned from all the data at once, as it already was with 0.1.0.
