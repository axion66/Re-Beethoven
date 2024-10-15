# Re-Beethoven

An attempt to regenerate Beethoven's works using deep learning.

# requirement

Need torch >= 2.4 for nn.RMSNorm, but maybe I can replace it w/ homemade

Since cost of Diffusion model is expensive, maybe using
time forecasting-like architecture(given time 0...T, output T+1...2T with transforms)
can be effective to demonstrate the output and then later try on diffusion + Discriminator.
