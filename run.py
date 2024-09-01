import sys
from nst import neural_style_transfer
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Neural Style Transfer")

    parser.add_argument("content_image_path", type=str, help="Path to the content image.")
    parser.add_argument("style_image_path", type=str, help="Path to the style image.")
    parser.add_argument("--style_layers", nargs='+', default=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'], help="List of style layers.")
    parser.add_argument("--content_layer", type=str, default="conv4_2", help="The content layer.")
    parser.add_argument("--style_layer_weights", nargs='+', type=float, default=[], help="Weights for each style layer.")
    parser.add_argument("--alpha", type=float, default=1, help="Content coefficient.")
    parser.add_argument("--beta", type=float, default=10, help="Style coefficient.")
    parser.add_argument("--gamma", type=float, default=0, help="Variation coefficient (gamma).")
    parser.add_argument("--num_iterations", type=int, default=50000, help="Number of iterations for optimization.")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate for optimization.")
    parser.add_argument("--save_every", type=int, default=None, help="Save the generated image every n iterations. Default is None.")

    return parser.parse_args()



if __name__ == '__main__':
    args = parse_arguments()
    
    generated_image, losses = neural_style_transfer(
        content_image_path=args.content_image_path,
        style_image_path=args.style_image_path,
        style_layers=args.style_layers,
        content_layer=args.content_layer,
        style_layer_weights=args.style_layer_weights,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
        save_every=args.save_every  # pass the new argument
    )