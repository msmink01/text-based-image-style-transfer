# Image Style Transfer Final Project

#### Final project for DSAIT4120: Applied Image Processing at TU-Delft. Authors: P. Johari, O. Milchi, & M. Smink

Each author implemented different style transfer components. The breakdown of these components per author can be seen in the project report.

## Demo

Link to demo on YouTube can be found [here](https://youtu.be/MqrvMC7lqj0).

[![Demo image](https://img.youtube.com/vi/MqrvMC7lqj0/0.jpg)](https://www.youtube.com/watch?v=MqrvMC7lqj0)

Or please see `AIP_FInalVideo.mp4` for a demo of this application and `DSAIT4120_FinalProjectReport.pdf` for a very basic explanation of the components in this project.

## Environment Setup Instructions

### Create Virutal Environment and Install Necessary Packages
```
$ conda create -n 4120 python=3.10
$ conda activate 4120
# If you want GPU enabled:
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# If you are okay with potentially no GPU:
$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.6 -c pytorch
$ pip install -r requirements.txt
```

### Download/Unzip Model files

1. Go to [model zip file](https://drive.google.com/file/d/1HLxFaeMXduE6tWcNpbJfdHFTO2dy88Bn/view?usp=sharing) and download the file.
2. Unzip file and move models into `text/subnetworks/checkpoints/`.

#### Final `text/` directory should look like:
```
text/
  Noto_Color_Emoji/
    ...
  subnetworks/
    ...
    checkpoints/
      clip_text_embedding_transformer.pth
      groundingdino_swint_ogc.pth
      GroundingDINO_SwinT_OGC.py
      image_transformer.pth
      sam_vit_b_01ec64.pth
```

## Running the Application

1. Ensure you are in your virtual environment (Run `conda activate 4120` if not).
2. From the main directory of this repository, run `python app.py`

Once the application has booted, you should see a message like:
```
Time taken to load all models: X.X seconds
Running on local URL: http://127.0.0.1:7860

To create a public link, set 'share=True' in 'launch()'.
```

3. Ctrl+Click on the local URL created by gradio.

The application should now be open in your web browser.

## Where to Find Components in Repository

### Text Style Transfer

In `app.py`, text style transfer is implemented in lines 161-282.

Any underlying code used in `app.py` relating to text style transfer can be found in the `text/` directory. Specifically:
- Base text style transfer is implemented in `text/FastTextTransfer.py`
- Localized text style transfer is implemented in `text/TextMaskExtractor.py` and `text/segmenetation_style_transfer.py`
- Texture text style transfer is implemented in `text/EmojiMaskExtractor.py` and `text/emoji_segmenetation_style_transfer.py`.

Any pre-trained models used during text style transfer can be found in the `text/subnetworks` directory.

Text-based segmentation is integrated for every other personal component. The code for this integration can be seen in the apply_image_process() function in `app.py`.

### Video Style Transfer

In `app.py`, video style transfer is implemented in the apply_video_process() method. Video style transfer is integrated with every other personal component.

### Color Palette Transfer

Color Palette Transfer is integrated in the UI in lines 592-658 of `app.py`.
The underlying functionality of this component is included in `color_palette/ColorPaletteTransfer.py`

### Style Mixing

Style Mixing is integrated in the UI in lines 472-572 of `app.py`.
The underlying functionality of this component is included in `multi_style_transfer`:
- channel attention is implemented in `multi_style_transfer/ChannelAttention.py`
- the style mixing is implemented in `multi_style_transfer/StyleMixer.py`
- the rest of the files implement style transfer as in Assignment 3

### Pixel Art
Pixel art code in `app.py` is implemented by calling fuction at 368.
The underlying functionality of this component is included in `components/pixel_art/`:
- `pixel_art.py` contains the code for the class that generates pixel art from an image
- `colour_palette.py` contains the code for the class that generates a colour palette and related functions
- `utils.py` contains some common code 
- `pixel_art_refrence.png` is the reference image for the pixel art and how i want it to look like 
- `100.json` is the list of colour pallets combined from [repo](https://github.com/Experience-Monks/nice-color-palettes) and [repo](https://github.com/thiagodnf/color-palettes/blob/master/data/palettes.json)

### Depth Style Transfer
Depth style transfer code in `app.py` is implemented by calling fuction at 733.
The underlying functionality of this component is included in `components/depth_style_transfer/`:
- `depth_style_transfer.py` contains the code for the class that generates depth style transfer from an image
- 'style_a3.py' contains the code for the class that generates style transfer as in Assignment 3 but with depth loss modification
- `utils.py` contains some extra code like the loss functions etc.
