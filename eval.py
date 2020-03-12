import tensorflow as tf
import os

def main():
    pass
    
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    discriminator = srgan.SRGanDiscriminator(training=g_training, image_size=args.image_size)
    generator = srgan.SRGanGenerator(discriminator=discriminator, training=d_training, learning_rate=args.learning_rate,
                                     content_loss=args.content_loss, use_gan=args.use_gan)
    pass