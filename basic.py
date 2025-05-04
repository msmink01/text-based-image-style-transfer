import tkinter as tk
from tkinter import filedialog, Toplevel, ttk
from PIL import Image, ImageTk, ImageOps
import torch
import math
import numpy as np
from text.FastTextTransfer import FastTextStyleTransfer
from text.TextMaskExtractor import TextMaskExtractor
from text.EmojiMaskExtractor import EmojiMaskExtractor
from text.segmentation_style_transfer import segmentation_style_transfer
from text.emoji_segmentation_style_transfer import _merge_content_style_segmentation_masks, emoji_segmentation_style_transfer
#from style_transfer.run_style_transfer import run_style_transfer

#add stuff here, current if only accepts one image at a time
processing_options = [
    "Style Transfer", 
    "Grayscale", 
    "Depth Map Based Style Transfer", 
    "Text-Prompt Based Style Tranfer", # style transfer: no masking
    "Text-Prompt Based Location Masking", # content mask only
    "Text-Based Masked Style Transfer", # style transfer: content mask + text, simple edges
    "Emoji-Based Text Mask Generation", # style emoji mask only
    "Text-Prompt Emoji-Based Location Masking", # style augmented content mask only
    "Text-Prompt Emoji-Location-Based Masked Style Transfer", # style transfer: style augmented content mask
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

text_transfer_model = FastTextStyleTransfer(device)
mask_extractor = TextMaskExtractor(device)
emoji_mask_extractor = EmojiMaskExtractor(device)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Image Processor")
        self.geometry("400x400")
        self.create_widgets()

    def create_widgets(self):
        self.open_button = tk.Button(self, text="Open Image", command=self.open_file_dialog)
        self.open_button.pack(pady=10)

        paddings = {'padx': 5, 'pady': 5}
        self.processing_var = tk.StringVar(value="Open Image Processing Menu")
        self.processing_menu = tk.OptionMenu(self, self.processing_var, *processing_options, command=self.option_printer)
        self.processing_menu.pack(pady=10)
        self.output_label = ttk.Label(self, foreground='red')
        self.output_label.pack(pady=10)

        self.status_label = tk.Label(self, text="", fg="green")
        self.status_label.pack(pady=10)

        # Style text prompt input (usually hidden)
        self.style_text_prompt_label = ttk.Label(self, text="Enter your text style prompt:")
        self.style_text_prompt_label.pack(pady=5)
        self.style_text_prompt_entry=ttk.Entry(self)
        self.style_text_prompt_entry.insert(0, "fire")
        self.style_text_prompt_entry.pack(pady=5)
        self.style_text_prompt_label.pack_forget()
        self.style_text_prompt_entry.pack_forget()

        # Mask text prompt input (usually hidden)
        self.mask_text_prompt_label = ttk.Label(self, text="Enter your text mask prompt:")
        self.mask_text_prompt_label.pack(pady=5)
        self.mask_text_prompt_entry=ttk.Entry(self)
        self.mask_text_prompt_entry.insert(0, "boat")
        self.mask_text_prompt_entry.pack(pady=5)
        self.mask_text_prompt_label.pack_forget()
        self.mask_text_prompt_entry.pack_forget()

        # Edge smoothing checkbox (usually hidden)
        self.edge_smoothing_var = tk.BooleanVar(value=False)
        self.edge_smoothing_checkbox = tk.Checkbutton(self, text="Enable Edge Smoothing", variable=self.edge_smoothing_var)
        self.edge_smoothing_checkbox.pack(pady=5)
        self.edge_smoothing_checkbox.pack_forget()  # Hide initially

        # Style strength slider
        self.style_strength_label = ttk.Label(self, text="Style Strength:")
        self.style_strength_label.pack_forget()
        self.style_strength_slider = ttk.Scale(self, from_=0.0, to=8.0, orient="horizontal", command=self.update_slider_label)
        self.style_strength_slider.set(1.5)  # Default value
        self.style_strength_slider.pack_forget()  # Initially hidden

        self.style_mixing_sliders = []
        self.style_mixing_labels = []

    def option_printer(self, value):
        if value == "Style Transfer":
            self.output_label['text'] = "Please select Image and Style Image in the order"
            self.hide_style_text_prompt()
            self.hide_mask_style_prompt()
            self.edge_smoothing_checkbox.pack_forget()  # Hide checkbox
            self.hide_slider()
            self.hide_all_style_mixing_sliders()
        elif value == "Depth Map Based Style Transfer":
            self.output_label['text'] = "Please select Image, Depth Map and Style Image in the order"
            self.hide_style_text_prompt()
            self.hide_mask_text_prompt()
            self.edge_smoothing_checkbox.pack_forget()  # Hide checkbox
            self.hide_slider()
            self.hide_all_style_mixing_sliders()
        elif value == "Text-Prompt Based Style Tranfer":
            self.output_label['text'] = "Please type your style prompt and select an input image in that order"
            self.show_style_text_prompt()
            self.hide_mask_text_prompt()
            self.edge_smoothing_checkbox.pack_forget()  # Hide checkbox
            self.hide_slider()
            self.hide_all_style_mixing_sliders()
        elif value == "Text-Prompt Based Location Masking":
            self.output_label['text'] = "Please type your desired object to mask as a prompt and select an input image in that order"
            self.hide_style_text_prompt()
            self.show_mask_text_prompt()
            self.edge_smoothing_checkbox.pack_forget()  # Hide checkbox
            self.hide_slider()
            self.hide_all_style_mixing_sliders()
        elif value == "Text-Based Masked Style Transfer":
            self.output_label['text'] = "Please type your desired style and object to mask as a prompt and select an input image in that order"
            self.show_style_text_prompt()
            self.show_mask_text_prompt()
            self.edge_smoothing_checkbox.pack(pady=5)  # Show checkbox
            self.hide_slider()
            self.hide_all_style_mixing_sliders()
        elif value == "Emoji-Based Text Mask Generation":
            self.output_label['text'] = "Please type your desired style to create a mask for based on emojis"
            self.show_style_text_prompt()
            self.hide_mask_text_prompt()
            self.edge_smoothing_checkbox.pack_forget()  # Hide checkbox
            self.hide_slider()
            self.hide_all_style_mixing_sliders()
        elif value == "Text-Prompt Emoji-Based Location Masking":
            self.output_label['text'] = "Please type your desired style and object to mask as a prompt and select an input image in that order"
            self.show_style_text_prompt()
            self.show_mask_text_prompt()
            self.edge_smoothing_checkbox.pack_forget()  # Hide checkbox
            self.hide_slider()
            self.hide_all_style_mixing_sliders()
        elif value == "Text-Prompt Emoji-Location-Based Masked Style Transfer":
            self.output_label['text'] = "Please type your desired style and object to mask as a prompt and select an input image in that order"
            self.show_style_text_prompt()
            self.show_mask_text_prompt()
            self.edge_smoothing_checkbox.pack_forget()  # Hide checkbox
            self.show_slider()
            self.hide_all_style_mixing_sliders()
        elif value == "Grayscale":
            self.output_label['text'] = "Please select Image"
            self.hide_style_text_prompt()
            self.hide_mask_text_prompt()
            self.edge_smoothing_checkbox.pack_forget()  # Hide checkbox
            self.hide_slider()
            self.hide_all_style_mixing_sliders()
        elif value == "Mixed Style Transfer":
            self.output_label['text'] = "Please select Image and (one or) multiple Style Images in that order"
            self.hide_style_text_prompt()
            self.hide_mask_text_prompt()
            self.edge_smoothing_checkbox.pack_forget()  # Hide checkbox
            self.hide_slider()
        else:
            self.output_label['text'] = "No option selected"
            self.hide_style_text_prompt()
            self.hide_mask_text_prompt()
            self.edge_smoothing_checkbox.pack_forget()  # Hide checkbox
            self.hide_slider()
            self.hide_all_style_mixing_sliders()

    def hide_style_text_prompt(self):
        self.style_text_prompt_label.pack_forget()
        self.style_text_prompt_entry.pack_forget()

    def show_style_text_prompt(self):
        self.style_text_prompt_label.pack(pady=5)
        self.style_text_prompt_entry.pack(pady=5)

    def hide_mask_text_prompt(self):
        self.mask_text_prompt_label.pack_forget()
        self.mask_text_prompt_entry.pack_forget()

    def show_mask_text_prompt(self):
        self.mask_text_prompt_label.pack(pady=5)
        self.mask_text_prompt_entry.pack(pady=5)

    def hide_slider(self):
        self.style_strength_label.pack_forget()
        self.style_strength_slider.pack_forget()

    def hide_all_style_mixing_sliders(self):
        for slider in self.style_mixing_sliders:
            slider.pack_forget()

    def show_slider(self):
        self.style_strength_label.pack(pady=5)
        self.style_strength_slider.pack(pady=5)

    def show_all_style_mixing_sliders(self, no_of_sliders):
        for i in range(no_of_sliders):
            slider = ttk.Scale(self, from_=0.0, to=1.0, orient="horizontal")
            label = ttk.Label(self, text=f"Style Strength for Image {i}:")
            self.style_mixing_sliders.append(slider)
            self.style_mixing_labels.append(label)
        for slider in self.style_mixing_sliders:
            slider.pack(pady=5)
        for label in self.style_mixing_labels:
            label.pack(pady=5)


    def update_slider_label(self, val):
        # Update the style strength label based on the slider value
        self.style_strength_label.config(text=f"Style Strength: {math.floor(float(val) * 2 + 0.5) / 2}") # round to nearest 0.5

    def process_image_factory(self, process_type):
        def style_process_image(file_path):
            try:
                # Open the image
                img = Image.open(file_path[0])
                # Apply selected processing
                if process_type == "Style Transfer":
                    style_img = Image.open(file_path[1])
                    #TODO apply style transfer
                    processed_img = img
                elif process_type == "Depth Map Based Style Transfer":
                    depth_map = Image.open(file_path[1])
                    style_img = Image.open(file_path[2])
                    #TODO apply depth map based style transfer
                    processed_img = img
                elif process_type == "Text-Prompt Based Style Tranfer":
                    text_prompt = self.style_text_prompt_entry.get()
                    print(f"Performing Text-Prompt Based Style Transfer using text prompt: {text_prompt}")
                    processed_img = text_transfer_model.perform_transfer(img, text_prompt)
                elif process_type == "Text-Prompt Based Location Masking":
                    text_prompt = self.mask_text_prompt_entry.get()
                    print(f"Performing Text-Prompt Based Location Masking using text prompt: {text_prompt}")
                    mask = mask_extractor.perform_mask_extraction(file_path[0], text_prompt)
                    processed_img = Image.fromarray(mask)
                elif process_type == "Text-Based Masked Style Transfer":
                    edge_smoothing = self.edge_smoothing_var.get()  # Get the checkbox value
                    style_text_prompt = self.style_text_prompt_entry.get()
                    mask_text_prompt = self.mask_text_prompt_entry.get()
                    print(f"Performing Text Based Masked Style Transfer using text prompts: {style_text_prompt} and {mask_text_prompt} with edge smoothing = {edge_smoothing}")
                    mask = mask_extractor.perform_mask_extraction(file_path[0], mask_text_prompt)
                    processed_img = text_transfer_model.perform_transfer(img, style_text_prompt)
                    processed_img = segmentation_style_transfer(img, processed_img, mask, edge_smoothing=5 if edge_smoothing else 0)
                elif process_type == "Emoji-Based Text Mask Generation":
                    text_prompt = self.style_text_prompt_entry.get()
                    print(f"Performing Text-Prompt Emoji-Based Style Masking using text prompt: {text_prompt}")
                    mask = emoji_mask_extractor.perform_emoji_mask_extraction(text_prompt)
                    processed_img = Image.fromarray(mask.astype(np.uint8)*255)
                elif process_type == "Text-Prompt Emoji-Based Location Masking":
                    style_text_prompt = self.style_text_prompt_entry.get()
                    mask_text_prompt = self.mask_text_prompt_entry.get()
                    print(f"Performing Text Prompt Emoji-Based Location Masker using text prompts: {style_text_prompt} and {mask_text_prompt}")
                    mask = mask_extractor.perform_mask_extraction(file_path[0], mask_text_prompt)
                    emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_text_prompt)
                    merged_mask = _merge_content_style_segmentation_masks(mask, emoji_mask)
                    processed_img = Image.fromarray((merged_mask*255).astype(np.uint8))
                elif process_type == "Text-Prompt Emoji-Location-Based Masked Style Transfer":
                    style_text_prompt = self.style_text_prompt_entry.get()
                    mask_text_prompt = self.mask_text_prompt_entry.get()
                    style_strength = math.floor(float(self.style_strength_slider.get()) * 2 + 0.5) / 2  # Get the slider value rounded to 0.5
                    print(f"Performing Text-Prompt Emoji-Location-Based Masked Style Transfer using text prompts: {style_text_prompt} and {mask_text_prompt} with style strength {style_strength}")
                    mask = mask_extractor.perform_mask_extraction(file_path[0], mask_text_prompt)
                    emoji_mask = emoji_mask_extractor.perform_emoji_mask_extraction(style_text_prompt)
                    processed_img = text_transfer_model.perform_transfer(img, style_text_prompt)
                    processed_img = emoji_segmentation_style_transfer(img, processed_img, mask, emoji_mask, style_strength=style_strength)
                elif process_type == "Grayscale":
                    processed_img = ImageOps.grayscale(img)
                elif process_type == "Mixed Style Transfer":
                    style_images = []
                    for i in range(1, len(file_path)):
                        style_images.append(Image.open(file_path[i]))
                    self.show_all_style_mixing_sliders(len(style_images))
                else:
                    processed_img = img

                # Show the processed image in a new window
                self.show_image_in_new_window(processed_img, process_type)
            except Exception as e:
                if isinstance(e, IndexError):
                    self.status_label.config(text="Please select the required files for processing")
                self.status_label.config(text=f"Error processing image: {e}")
        return style_process_image

    def open_file_dialog(self):
        file_path = filedialog.askopenfilenames(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if len(file_path) >= 1:
            self.status_label.config(text=f"Files selected: {file_path}")
            process_type = self.processing_var.get()
            self.process_image_factory(process_type)(file_path)

    def show_image_in_new_window(self, image, title):
        def save_image():
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                                ("All files", "*.*")])
            if save_path:
                image.save(save_path)
                self.status_label.config(text=f"Image saved at: {save_path}")

        # Resize the image to fit within the window
        view_image = self.resize_image(image, 400, 400)


        new_window = Toplevel(self)
        new_window.title(f"Processed Image - {title}")
        new_window.geometry("400x400")

        img_tk = ImageTk.PhotoImage(view_image)
        img_label = tk.Label(new_window, image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection
        img_label.pack()

        save_button = tk.Button(new_window, text="Save Image", command=save_image)
        save_button.pack(pady=10)

    def resize_image(self, image, max_width, max_height):
        # Get the current size of the image
        width, height = image.size

        # Calculate the scaling factor to fit the image within the max width and height
        scaling_factor = min(max_width / width, max_height / height)

        # If the image is smaller than the max size, no need to resize
        if scaling_factor > 1:
            return image

        # Calculate the new dimensions while maintaining the aspect ratio
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        # Resize the image
        resized_image = image.resize((new_width, new_height))
        return resized_image

if __name__ == "__main__":
    app = App()
    app.mainloop()