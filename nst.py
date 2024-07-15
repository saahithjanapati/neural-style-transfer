import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import io
import torch.nn as nn
from tqdm import tqdm


def generate_image_tensor(image_path):
    image_tensor = io.read_image(image_path)
    return image_tensor


layer_name_to_idx = {
    'conv1_1': 0,
    'conv1_2': 2,
    'conv2_1': 5,
    'conv2_2': 7,
    'conv3_1': 10,
    'conv3_2': 12,
    'conv3_3': 14,
    'conv3_4': 16,
    'conv4_1': 19,
    'conv4_2': 21,
    'conv4_3': 23,
    'conv4_4': 25,
    'conv5_1': 28,
    'conv5_2': 30,
    'conv5_3': 32,
    'conv5_4': 34}

result_dict = {}


# content loss
def content_loss(content_features, gen_features):
    return 0.5 * F.mse_loss(content_features, gen_features)


# style loss
def gram_matrix(feature_map):
    if feature_map.dim() == 4:  # make sure the feature map is 3 dimensional (remove the extra batch dim)
        feature_map = feature_map.squeeze(0)

    num_channels, width, height = feature_map.size()
    flat_feature_map = feature_map.view(num_channels, width * height)
    return flat_feature_map @ flat_feature_map.T


def style_loss(style_features, gen_features, layer_weights):
    loss = 0

    for style_feature, gen_feature, layer_weight in zip(style_features, gen_features, layer_weights):
        # compute gram matrix of the style features
        style_gram = gram_matrix(style_feature)
        gen_gram = gram_matrix(gen_feature)
        
        
        
        _, num_channels, width, height = style_feature.shape
        loss += F.mse_loss(style_gram, gen_gram) / (4 * num_channels**2 * width * height) * layer_weight

    return loss


def total_variation_loss(image):
    shift_to_right = image[:, :, :, 1:]
    shift_up = image[:, :, 1:, :]
    loss = 0
    loss += F.mse_loss(image[:, :, :, :-1], shift_to_right)
    loss += F.mse_loss(image[:, :, :-1, :], shift_up)

    return loss


def create_forward_hook(layer_type, layer):
    def forward_hook(model, input, output):
        if layer_type not in result_dict:  # make sure image type dict exists
            result_dict[layer_type] = {}
        result_dict[layer_type][layer] = output

    return forward_hook


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def neural_style_transfer(content_image_path,
                          style_image_path,
                          style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                          content_layer="conv4_2",
                          style_layer_weights=[],
                          alpha=1e-2,
                          beta=1.0,
                          gamma=0,
                          num_iterations=1000,
                          learning_rate=1):

    device = get_best_device()

    if style_layer_weights == []:
        style_layer_weights = [1.0 for i in range(len(style_layers))]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize the image to the required input size of VGG19
        transforms.ToTensor(),  # Convert PIL image or numpy array to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load up the style image as a tensor
    style_image_tensor = generate_image_tensor(style_image_path)
    style_image_tensor = transform(style_image_tensor).unsqueeze(0).to(device).detach()

    # load up the content image as a tensor
    content_image_tensor = generate_image_tensor(content_image_path)
    content_image_tensor = transform(content_image_tensor).unsqueeze(0).to(device).detach()

    # load up the vgg model
    vgg19_model = models.vgg19(pretrained=True).to(device).eval()

    for param in vgg19_model.parameters():
        param.requires_grad = False

    hook_handles = []

    # register the hooks onto the vgg19_model
    for layer_name in style_layers:
        layer_idx = layer_name_to_idx[layer_name]
        hook_func = create_forward_hook("style", layer_idx)

        hook = vgg19_model.features[layer_idx].register_forward_hook(hook_func)
        hook_handles.append(hook)

    content_layer_idx = layer_name_to_idx[content_layer]
    content_hook_func = create_forward_hook("content", content_layer_idx)

    handle = vgg19_model.features[content_layer_idx].register_forward_hook(content_hook_func)
    hook_handles.append(handle)

    # perform forward pass of the content image
    _ = vgg19_model(content_image_tensor)

    content_features = result_dict["content"][content_layer_idx]

    result_dict.clear()

    _ = vgg19_model(style_image_tensor)
    style_image_tensors = []

    for layer_name in style_layers:
        layer_idx = layer_name_to_idx[layer_name]
        style_image_tensors.append(result_dict['style'][layer_idx])

    generated_image = nn.Parameter(content_image_tensor.clone(), requires_grad=True)  # start the generated image from the content image
    optimizer = torch.optim.SGD([generated_image], lr=learning_rate)

    frames = []
    losses = []

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(3, 1, 1)

    # optimization loop
    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad()

        _ = vgg19_model(generated_image)

        gen_content_feature = result_dict["content"][content_layer_idx]
        gen_style_tensors = []

        for layer_name in style_layers:
            layer_idx = layer_name_to_idx[layer_name]
            gen_style_tensors.append(result_dict['style'][layer_idx])

        # style loss
        style_loss_val = style_loss(style_image_tensors, gen_style_tensors, style_layer_weights)

        # content loss
        content_loss_val = content_loss(content_features, gen_content_feature)

        # total variation loss
        variation_loss_val = total_variation_loss(generated_image)

        total_loss = style_loss_val * beta + content_loss_val * alpha + variation_loss_val * gamma

        losses.append(total_loss.item())

        total_loss.backward()
        optimizer.step()

        result_dict.clear()

        frame_copy = generated_image.clone().squeeze(0).detach()
        frame_copy = frame_copy * std + mean
        frame_copy = frame_copy.to('cpu')
        frames.append(frame_copy)

    for hook_handle in hook_handles:
        hook_handle.remove()

    generated_image = generated_image * std + mean
    final_frame = generated_image.squeeze(0).to('cpu').detach()
    frames.append(final_frame)

    return final_frame, frames, losses