import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class EmojiMaskExtractor():
    def __init__(self, device='cuda', max_input_length=64, max_target_length=64):
        """
        Create an EmojiMaskExtractor object by creating the underlying T5-Base model.

        Used to get a mask for a text prompt with no input image needed.

        Args:
            device (str): device to use for the subnetworks: 'cuda' or 'cpu'
            max_input_length (int): max number of tokens to input as prompt
            max_target_length (int): max number of tokens to output from model
        """
        self.device = device
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        EMOJI_PATH = "KomeijiForce/t5-base-emojilm"

        print("Loading Emoji AutoTokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(EMOJI_PATH)

        print("Loading Emoji AutoModelForSeq2SeqLM...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(EMOJI_PATH).to(device)

    def perform_emoji_mask_extraction(self, text_prompt, prefix="translate to a single emoji:"):
        """
        Given an input text prompt extract a singular mask for the text prompt based on an emoji

        Args:
            text_prompt (str): object to extract mask for (Example: "fire")
            prefix (str): string prompt to prepend to text_prompt before being given to model

        Returns:
            mask (numpy array): True/False mask of where in image object appears of shape HxW
        """
        # Create text prompt
        inputs = [prefix + text_prompt]

        # Create inputs to model by tokenizing text prompt
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True, return_tensors="pt")
        input_ids = model_inputs.input_ids.to(self.model.device)
        attention_mask = model_inputs.attention_mask.to(self.model.device)

        # Get EmojiT5-base logits
        outputs = self.model.generate(input_ids, attention_mask=attention_mask,min_length = 1, max_length = self.max_target_length, do_sample=True, top_p=0.95, top_k=10)
        
        # Decode model output to text
        outputs_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # If no text returned output an empty mask
        if not len(outputs_str[0]) > 0: # ensure there is something returned
            return np.full([172,172,3], False)

        # Otherwise, get first emoji
        single_emoji = outputs_str[0][0]

        # Create img based on emoji
        blank_img = np.ones([172,172,3], dtype=np.uint8)*255
        pil_img = Image.fromarray(blank_img)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("./text/Noto_Color_Emoji/NotoColorEmoji_WindowsCompatible.ttf", size=109)
        draw.text((20,10), single_emoji, (0,0,0), font=font)

        # Convert emoji img to a mask
        emoji_array = np.array(pil_img)
        mask = np.where(emoji_array < 255, True, False)[:, :, 0]

        return mask