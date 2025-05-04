import copy
from multi_style_transfer.ChannelAttention import ChannelAttention
from multi_style_transfer.style_transfer_losses import *

def PIL_to_tensor(image):
    """"
    Transform a PIL image to a Tensor.
    """
    img = transforms.ToTensor()(image)
    img = img.unsqueeze(0)
    return img

def channel_att_per_chosen_layers(feats, layers, device):

    result = copy.deepcopy(feats)

    for layer in layers:
        ca = ChannelAttention(feats[layer].shape[1]).to(device)

        # Cast al layers' features to the same device
        feature = feats[layer]
        feature.to(device)
        result[layer] = ca(feature)

    return result

def run_multi_style_transfer(vgg_mean, vgg_std, content_img, num_steps, random_init, w_style, w_content, w_tv,
                       w_edge, style_img1, style_img2=None, style_img_weight=0.5, print_iter=50, channel_attention=False, device="cpu"):
    """ Neural Style Transfer optmization procedure for a single style image.
        NOTE: Implementation is taken from Assignment 3. Changes are made in style_transfer_losses.py (for content loss
        and style loss)
    # Parameters:
        @vgg_mean, VGG channel-wise mean, torch.tensor of size (c)
        @vgg_std, VGG channel-wise standard deviation, detorch.tensor of size (c)
        @content_img, content image as PIL image
        @style_img1, style image 1 as PIL image
        @style_img2, style image 2 as PIL image
        @num_steps, int, iteration steps
        @random_init, bool, whether to start optimizing with based on a random image. If false,
            the content image is as initialization.
        @w_style, float, weight for style loss
        @w_content, float, weight for content loss
        @w_tv, float, weight for total variation loss
        @w_edge, float, weight for edge loss
        @print_iter, int, iteration interval for printing the losses
        @channel_attention, bool, whether to use channel-wise attention when computing content loss
        @device, indicates whether to run on a CPU or GPU

    # Returns the style-transferred image
    """

    seed_everything(101)

    # Choose what feature maps to extract for the content and style loss
    # We use the ones as mentioned in Gatys et al. 2016
    content_layers = ['conv4_2']
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    content_img = PIL_to_tensor(content_img).to(device)
    style_img1 = PIL_to_tensor(style_img1).to(device)
    if style_img2:
        style_img2 = PIL_to_tensor(style_img2).to(device)
        normed_style_imgs = [normalize(style_img1, vgg_mean, vgg_std), normalize(style_img2, vgg_mean, vgg_std)]
    else:
        normed_style_imgs = [normalize(style_img1, vgg_mean, vgg_std)]

    # Initialize Model
    model = Vgg19(content_layers, style_layers, device)

    # TODO: 1. Normalize Input images
    normed_content_img = normalize(content_img, vgg_mean, vgg_std)

    if w_edge > 0:
        target_gradient_img = get_gradient_imgs(to_grayscale(normed_content_img)).detach()

    # Retrieve feature maps for content and style image
    # We do not need to calculate gradients for these feature maps
    with torch.no_grad():
        style_features = [model(normed_style_img) for normed_style_img in normed_style_imgs]
        content_features = model(normed_content_img)

    # Either initialize the image from random noise or from the content image
    if random_init:
        optim_img = torch.randn(content_img.data.size(), device=device)
        optim_img = torch.nn.Parameter(optim_img, requires_grad=True)
    else:
        optim_img = torch.nn.Parameter(content_img.clone(), requires_grad=True)

    # Initialize optimizer and set image as parameter to be optimized
    optimizer = optim.LBFGS([optim_img])

    print("Channel attention enabled: " + str(channel_attention))

    if channel_attention:
        content = channel_att_per_chosen_layers(content_features, content_layers, device=device)
    else: content = content_features

    # Training Loop
    iter = [0]
    while iter[0] <= num_steps:

        def closure():

            # Set gradients to zero before next optimization step
            optimizer.zero_grad()

            # Clamp image to lie in correct range
            with torch.no_grad():
                optim_img.clamp_(0, 1)

            # Retrieve features of image that is being optimized
            normed_img = normalize(optim_img, vgg_mean, vgg_std)
            input_features = model(normed_img)

            loss = torch.tensor([0.], device=device)
            # TODO: 2. Calculate the content loss
            c_loss = 0
            if w_content > 0:
                c_loss = w_content * content_loss(input_features, content, content_layers)

            # TODO: 3. Calculate the style loss
            s_loss = 0
            if w_style > 0:
                s_loss = w_style * style_loss(input_features, style_features, style_layers, style_img_weight)

            # TODO: 4. Calculate the total variation loss
            tv_loss = 0
            if w_tv > 0:
                tv_loss = w_tv * total_variation_loss(normed_img)

            e_loss = 0
            if w_edge > 0:
                # TODO: 5. Calculate the gradient images based on the sobel kernel
                gradient_optim = get_gradient_imgs(to_grayscale(optim_img))
                # TODO: 6. Calculate the edge loss
                e_loss = w_edge * edge_loss(target_gradient_img, gradient_optim)

            # Sum up the losses and do a backward pass
            loss = loss + s_loss + c_loss + tv_loss + e_loss
            loss.backward()

            # Print losses every 50 iterations
            iter[0] += 1
            if iter[0] % print_iter == 0:
                print(
                    f'Reached iteration {iter[0]}/{num_steps}')

            return loss

        # Do an optimization step as defined in our closure() function
        optimizer.step(closure)

    # Final clamping
    with torch.no_grad():
        optim_img.clamp_(0, 1)

    optim_img = transforms.ToPILImage()(optim_img.cpu().clone().squeeze(0))

    return optim_img
