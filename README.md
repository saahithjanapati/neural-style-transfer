# Neural Style Transfer

This repository is a PyTorch implementation of [Neural Style Transfer](https://arxiv.org/pdf/1508.06576)

![Sample Generated Image](generated_image.png)

## Dependencies

To install the necessary dependencies, you can use `pip`:

pip install torch torchvision tqdm Pillow

## Running the Algorithm



### Basic Usage

You can run the neural style transfer algorithm using the `run.py` script.

To run the algorithm with the default parameters, use the following command:

```python run.py content_image.jpg style_image.jpg```

The sample image above was generated with the default parameters.

### Custom Parameters

You can also customize the parameters by using the available options:

```
python run.py content_image.jpg style_image.jpg \
    --style_layers conv1_1 conv2_1 conv3_1 conv4_1 conv5_1 \
    --content_layer conv4_2 \
    --style_layer_weights 1.0 0.8 0.5 0.3 0.1 \
    --alpha 1e-1 \
    --beta 10 \
    --gamma 1e-4 \
    --num_iterations 40000 \
    --learning_rate 0.01 \
    --save_every 1000
```

### Argument Descriptions

- content_image_path: Path to the content image (required).
- style_image_path: Path to the style image (required).
- --style_layers: List of layers used for style representation. Default is ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'].
- --content_layer: The layer used for content representation. Default is conv4_2.
- --style_layer_weights: Weights for each style layer. Default is an empty list, meaning all layers are weighted equally.
- --alpha: Content coefficient. Default is 1e-1.
- --beta: Style coefficient. Default is 10.
- --gamma: Variation coefficient. Default is 1e-4.
- --num_iterations: Number of iterations for optimization. Default is 40000.
- --learning_rate: Learning rate for optimization. Default is 0.01.
- --save_every: Save the generated image every n iterations. Default is None.
