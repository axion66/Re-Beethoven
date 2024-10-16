# Re-Beethoven

An attempt to regenerate Beethoven's works using deep learning.

# requirement

Need torch >= 2.4 for nn.RMSNorm, but maybe I can replace it w/ homemade

Since cost of Diffusion model is expensive, maybe using
time forecasting-like architecture(given time 0...T, output T+1...2T with transforms)
can be effective to demonstrate the output and then later try on diffusion + Discriminator.

# experiment

using forecasting doesn't work as shown in withoutDiffusion.png
Although loss per epoch seems decreasing, its very small decreasing rate and oscillating loss measured per steps show it's not going to make it.
Even my intution says it doesn't seem promising, since every songs have different patterns, using MSE for convergence cannot be happened as "There is no formula or common for the next seconds."
Let's use GANs + Diffusion.
Now im just reading FLUX for better understanding of this diffusion world.
