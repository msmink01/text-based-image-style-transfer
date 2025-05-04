import torch
from transformers import pipeline
import torch.optim as optim

from components.style_transfer_depth.util import *


class StyleA3:
    """
    Neural Style Transfer implementation with depth and edge-aware enhancements.

    Attributes:
        device (str): Computation device ('cuda'/'cpu')
        content_layers (list): VGG layers for content representation
        style_layers (list): VGG layers for style representation
        depth_pipeline: Depth estimation model pipeline
    """
    def __init__(self, device, print_iter=50, num_steps=400, w_style=5e5, w_content=1, w_tv=2e1, w_edge=2e1, w_depth=0, random_init=False):
        """
        Initialize style transfer parameters and models.

        Args:
            device (str): Computation device ('cuda' or 'cpu')
            print_iter (int): Loss printing interval
            num_steps (int): Optimization iterations
            w_style (float): Style loss weight
            w_content (float): Content loss weight
            w_tv (float): Total variation loss weight
            w_edge (float): Edge preservation loss weight
            w_depth (float): Depth consistency loss weight
            random_init (bool): Initialize from random noise if True
        """

        self.device = device  # 'cuda' or 'cpu'
        self.print_iter = print_iter
        self.num_steps = num_steps
        self.w_style = w_style
        self.w_content = w_content
        self.w_tv = w_tv
        self.w_edge = w_edge
        self.w_depth = w_depth
        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.vgg_std = torch.tensor([0.485, 0.224, 0.225]).to(self.device)
        self.random_init = random_init
        self.depth_pipeline = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")


    def _get_depth_map(self, image):
        """
        Generate depth map using Depth-Anything model.

        Args:
            image (PIL.Image): Input image
        Returns:
            PIL.Image: Depth map visualization
        """
        depth = self.depth_pipeline(image)["depth"]
        return Image.fromarray(np.asarray(depth))

    def _run_style_transfer(self, style, content):
        """
        Core style transfer optimization process.

        Args:
            style (PIL.Image): Style reference image
            content (PIL.Image): Content source image
        Returns:
            tuple: (optimized image tensor, edge gradient tensor)
        """

        style_img = image_loader(style, device=self.device)
        content_img = image_loader(content, device=self.device)

        # Initialize Model
        model = Vgg19(self.content_layers, self.style_layers, self.device)


        normed_style_img = normalize(style_img, self.vgg_mean, self.vgg_std)
        normed_content_img = normalize(content_img, self.vgg_mean, self.vgg_std)

        if self.w_depth > 0:
            depth_img = image_loader(self._get_depth_map(content), device=self.device)
            normed_depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
            target_depth_img = depth_img.detach()

        if self.w_edge > 0:
            target_gradient_img = get_gradient_imgs(to_grayscale(normed_content_img)).detach()

        # Retrieve feature maps for content and style image
        # We do not need to calculate gradients for these feature maps
        with torch.no_grad():
            style_features = model(normed_style_img)
            content_features = model(normed_content_img)

        # Either initialize the image from random noise or from the content image
        if self.random_init:
            optim_img = torch.randn(content_img.data.size(), device=self.device)
            optim_img = torch.nn.Parameter(optim_img, requires_grad=True)
        else:
            optim_img = torch.nn.Parameter(content_img.clone(), requires_grad=True)

        # Initialize optimizer and set image as parameter to be optimized
        optimizer = optim.LBFGS([optim_img])

        # Training Loop
        iter = [0]
        while iter[0] <= self.num_steps:

            def closure():

                # Set gradients to zero before next optimization step
                optimizer.zero_grad()

                # Clamp image to lie in correct range
                with torch.no_grad():
                    optim_img.clamp_(0, 1)

                # Retrieve features of image that is being optimized
                normed_img = normalize(optim_img, self.vgg_mean, self.vgg_std)
                input_features = model(normed_img)

                loss = torch.tensor([0.], device=self.device)
                c_loss = torch.tensor([0.], device=self.device)
                if self.w_content > 0:
                    c_loss = self.w_content * content_loss(input_features, content_features, self.content_layers)

                s_loss = torch.tensor([0.], device=self.device)
                if self.w_style > 0:
                    s_loss = self.w_style * style_loss(input_features, style_features , self.style_layers)

                tv_loss = torch.tensor([0.], device=self.device)
                if self.w_tv > 0:
                    tv_loss = self.w_tv * total_variation_loss(normed_img)

                e_loss = torch.tensor([0.], device=self.device)
                if self.w_edge > 0:
                    gradient_optim = get_gradient_imgs(to_grayscale(optim_img))
                    e_loss = self.w_edge * edge_loss(target_gradient_img, gradient_optim)

                d_loss = torch.tensor([0.], device=self.device)
                if self.w_depth > 0:
                    depth_optim = image_loader(self._get_depth_map(save_image(optim_img)), device=self.device)
                    depth_optim = (depth_optim - depth_optim.min()) / (depth_optim.max() - depth_optim.min())
                    d_loss = self.w_depth * depth_loss(depth_optim, target_depth_img)

                # Sum up the losses and do a backward pass
                loss = loss + s_loss + c_loss + tv_loss + e_loss + d_loss
                loss.backward()

                # Print losses every 50 iterations
                iter[0] += 1
                if iter[0] % self.print_iter == 0:
                    print(
                        f'iter {iter[0]}: Content Loss: {c_loss.item():4f} | Style Loss: {s_loss.item():4f} | TV Loss: {tv_loss.item():4f} | Edge Loss: {e_loss.item():4f} | Depth Loss: {d_loss.item():4f} | Total Loss: {loss.item():4f} ')
                return loss

            # Do an optimization step as defined in our closure() function
            optimizer.step(closure)

        # Final clamping
        with torch.no_grad():
            optim_img.clamp_(0, 1)

        return optim_img, target_gradient_img

    def style_transfer(self, style, content, depth=False, strength=1):
        """
        Public interface for style transfer execution.

        Args:
            style (PIL.Image): Style reference image
            content (PIL.Image): Content source image
            depth (bool): Enable depth-aware loss
            strength (float): Style influence multiplier
        Returns:
            PIL.Image: Stylized output image
        """
        if depth:
            self.w_depth = 5e4
        else:
            self.w_depth = 0
        if strength <0:
            self.w_style = 5e5
        else:
            self.w_style = 5e5 * (np.e**(strength - 1/strength))
        print(f"w_style: {self.w_style}")
        print(f"Style Transfer with strength {strength}")
        optim_img, _ = self._run_style_transfer(style, content)

        return save_image(optim_img)

