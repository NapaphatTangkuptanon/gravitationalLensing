"# gravitationalLensing" 

This is Computational Physic project.

1.Solve equation 3 on the grid: use 5-point stencil with Gauss–Seidel / SOR iteration. Apply Dirichlet boundary condition  = 0 at the edges. (L=6 , N= 512, pixel size =L/N, number of iteration is in optimization progress)
2.Compute deflection and shear: deflection from equation 2, shear from second derivative.
3.Ray Tracing to form image: map each image plane from the source plane equation.
4.Instrument model: Applied Gaussian PSF convolution (FFT), Added Poisson and Gaussian read noise to generate noisy image.
5. Simulate and visualize. Produce clearly labeled plots with colorbars:
(a) Convergence κ(θx, θy ),
(b) Potential ψ (contours),
(c) Deflection magnitude ∥α∥,
(d) Magnification μ (clip extreme values for display),
(e) Unlensed source S(β),
(f) Final noisy lensed image (arcs/ring).
6. Measurements. Estimate the Einstein-ring radius from the lensed image
and compare with the input θE . Report the total magnification
M =
P I(θ)
P S(β)
using the same field of view. Identify multiple images and mark their
positions.
7. Parameter study. Vary γ ∈ {0, 0.05, 0.1, 0.2} and the source offset
∥β0∥ ∈ {0, 0.1′′, 0.2′′}. For each case, record ring/arc morphology, mea-
sured ring radius, and total magnification. Summarize trends in a small
table and 2–3 sentences.
8.Fermat surface and time delays. Plot the Fermat
potential
mark stationary points (image locations), and discuss minima vs. saddles.

This is what have been done by this respiratory

Suggestion:

1.Dont run GuassSeidal.py since it's took very long time to run
"
1.Stimulate and visualize: extremely long execution time
2.Measurement: estimate Einstein-ring radius, compute total magnification.
3.Parameter study: adjust and visualize how the changes in parameters affect the image.

"