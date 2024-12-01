# Re-Beethoven

An attempt to regenerate Beethoven's works using deep learning.


# experiment

Loss:
    AuraLoss, GAN Wrapper(EncodecDiscriminator in stable audio tools)

Model (Uncond):
    Encoder + DiT + Decoder

    Autoencoder = Encoder + BottleNeck(VAE,AE) + Decoder
        trained before and freezed for Diffusion 


Dataset:
    Train: Musicnet(Kaggle)
    Debug: 3~5 musics (a collective set w/ Public domains)

