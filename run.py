import sys
import matplotlib.pyplot as plt
import imageio
from nst import neural_style_transfer
from PIL import Image
import numpy as np

def save_image(tensor, path):
    image = tensor.permute(1, 2, 0).numpy()
    image = (image * 255).astype('uint8')
    Image.fromarray(image).save(path)

def create_animation(frames, losses, gif_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    def update(frame):
        ax1.clear()
        ax2.clear()
                
        ax1.imshow(frame.permute(1, 2, 0).numpy())
        ax1.axis('off')
        
        ax2.plot(losses[:len(frames)])
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Curve')
    
    with imageio.get_writer(gif_path, mode='I', duration=0.005) as writer:
        for frame in frames:
            update(frame)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python run.py <content_image_path> <style_image_path>")
        sys.exit(1)
    
    content_image_path = sys.argv[1]
    style_image_path = sys.argv[2]
    
    generated_image, frames, losses = neural_style_transfer(content_image_path, style_image_path)
    
    print(generated_image.shape)
    print(frames[0].shape)
    print(losses)
    save_image(generated_image, 'generated_image.png')
    # create_animation(frames, losses, 'style_transfer.gif')