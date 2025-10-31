| Dataset        | Type             | Expression |
|----------------|------------------|-------------|
| S-Shape        | Rectangle        | $0.3 \leq x \leq 1, 0.55 \leq y \leq 0.65$ |
| C-Shape        | Custom           | $(\frac{x+0.3}{6})^{\frac{2}{3}} + (\frac{y-0.4}{3})^{\frac{2}{3}} \leq 0.15$ |
| Worm           | Rectangle        | $-0.4 \leq x \leq -0.3, -0.1 \leq y \leq 0.05$ |
| Angle          | Rectangle        | $-0.6 \leq x \leq -0.4, 0.7 \leq y \leq 1.0$ |
| G-Shape        | Circle           | $x^2 + (y-0.25)^2 \leq 0.01$ |
| P-Shape        | Circle           | $x^2 + (y-0.4)^2 \leq 0.04$ |
| N-Shape        | Custom           | $\max(y,-4x-0.8y-1.6,-0.8y+4x+0.8) \leq 0$ |
| Sine-NEW       | Custom           | $\max(y + 0.2\sin(9.5x - \pi/6) - 0.5, -0.2\sin(9.5x - \pi/6) + 0.28 - y, x, -1-x) \leq 0$ |
| 3D C-Shape     | Sphere           | $(x-0.4)^2 + (y-0.2)^2 + (z+0.3)^2 \leq 0.0625$ |
| Robot Dataset  | Multiple Circles | $(x-0.25)^2 + (y+0.2)^2 \leq 0.0036, (x-0.8)^2 + (y+0.2)^2 \leq 0.0036, (x-0.5)^2 + (y+0.55)^2 \leq 0.0036$ |
