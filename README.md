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

SDE vs ODE, (w/ rectified flow)
treating audio as image (with pretrained models i.g. Flux that plays music) or treating audio with STFT or some latent codebook.

-> decided to discard parallel LSTM cuz it's not proven that it's more effective than current transformers.
-> Add MSD @ MPD for final outputs.
-> diffusion is essential, but not sure what architecture(https://arxiv.org/pdf/2206.00364)
-> somewhat we stuck at 16khz with ~10 to ~30 sec audios. (prolly due to limited available datasets, so i should find more on myself??)
-> want non-text-conditional as then it's gonna take lots of time.
-> Other boring stuffs: EMA, gradient clipping, hyperparameters.
-> loss?? -> use diffusion backpropagation(E(Noise) - noise) and discriminator loss,
but found some papers use stft_loss(loss based on magnitude), -> found out that only TTS or Text-to-video model uses it... so dumb. (makes sence as they need some alingments)
-> SR comparison: 8000 works pretty fine, <4000hz seems not high-quality
-> Should consider using WavTokenizer?
