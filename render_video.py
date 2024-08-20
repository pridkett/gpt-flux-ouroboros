import argparse
import json
import os
import tempfile
import shutil
import logging
from PIL import Image, ImageDraw, ImageFont

from rich.logging import RichHandler
from rich.traceback import install

from typing import Tuple, List, TypedDict

install()

INTERMEDIATE_FRAMES = 30
FRAME_RATE = 30
STILL_FRAMES = 15
BITRATE = "5M"
CRF = 18
FONT_NAME = "ariblk.ttf"
FONT_SIZE = 32
BOTTOM_PADDING = 20

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("run_workflow")
log.setLevel(logging.DEBUG)

class TextLineDict(TypedDict):
    """
    Represents a dictionary containing information about a line of text for render_text.

    Attributes:
        width (int): The width of the text line.
        height (int): The height of the text line.
        text (str): The actual text content of the line.
    """
    width: int
    height: int
    text: str

class BlurryDict(TypedDict):
    value: bool
    variance: float
    
class ImageDict(TypedDict):
    blurry: BlurryDict
    NIMA_score: float
    filename: str

class FrameDict(TypedDict):
    run_id: str
    run_start_time: str
    seed: int
    source_image: str
    ctr: int
    previous_image: str
    last_failed: bool
    variance_threshold: int
    sbs_images: List[ImageDict]
    images: List[ImageDict]
    best_image: ImageDict
    failed: bool
    llm_prompt: List[str]
    time: str


def get_text_size(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    """
    Calculate the size of the rendered text.

    Args:
        text (str): The text to be rendered.
        draw (ImageDraw.ImageDraw): The ImageDraw object used for rendering.
        font (ImageFont.ImageFont): The font used for rendering.

    Returns:
        Tuple[int, int]: A tuple representing the width and height of the rendered text.
    """
    (left, top, right, bottom) = draw.textbbox((0, 0), text, font=font)
    log.debug(f"bbox: {left}, {right}, {top}, {bottom}")
    return (right-left, bottom-top)

def render_text(image: Image.Image, text: str, font_name: str, font_size: int, bottom_padding: int = BOTTOM_PADDING) -> Image.Image:
    """
    Renders the given text onto the image using the specified font and size.
    Args:
        image (Image.Image): The image onto which the text will be rendered.
        text (str): The text to be rendered.
        font_name (str): The name of the font to be used.
        font_size (int): The size of the font.
        bottom_padding (int, optional): The amount of padding at the bottom of the image. Defaults to BOTTOM_PADDING.
    Returns:
        Image: The image with the rendered text.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_name, font_size)
    image_width, image_height = image.size

    output_text: List[TextLineDict] = []
    input_text = text.splitlines()

    # iterate through the lines of text and add in wrapping as necessary
    for input_line in input_text:
        words = input_line.split(" ")
        output_words: List[str] = []
        text_width, text_height = 0, 0
        for word in words:
            test_line = " ".join(output_words + [word])
            text_width, text_height = get_text_size(test_line, draw, font)
            if text_width > image_width:
                output_text.append({"width": text_width, "height": text_height, "text": " ".join(output_words)})
                output_words = [word]
            else:
                output_words = output_words + [word]
        output_text.append({"width": text_width, "height": text_height, "text": " ".join(output_words)})
    
    # iterate through the text starting from the bottom and render it centered
    text_y = image_height - bottom_padding
    for output_line in output_text[::-1]:
        text_y -= output_line["height"]
        text_x = (image_width - output_line["width"]) // 2
        draw.text((text_x, text_y), text=output_line["text"], font=font) # type: ignore

        # Draw black outline
        outline_width = 2
        for x in range(-outline_width, outline_width + 1):
            for y in range(-outline_width, outline_width + 1):
                draw.text((text_x + x, text_y + y), output_line["text"], font=font, fill="black") # type: ignore

        # Draw white fill
        draw.text((text_x, text_y), output_line["text"], font=font, fill="white") # type: ignore
        
    return image

def main(json_file: str, image_dir: str, output_file: str, interemediate_frames: int=INTERMEDIATE_FRAMES, frame_rate: int=FRAME_RATE,
         bitrate: str=BITRATE, crf: int=CRF, still_frames: int=STILL_FRAMES,
         font_name: str=FONT_NAME, font_size: int=FONT_SIZE) -> None:
    """
    Render a video using the provided JSON file and image directory.
    Args:
        json_file (str): The path to the JSON file containing the run log.
        image_dir (str): The directory containing the frame images.
        output_file (str): The path to the output video file.
        interemediate_frames (int, optional): The number of intermediate frames to generate between each pair of frames. Defaults to INTERMEDIATE_FRAMES.
        frame_rate (int, optional): The frame rate of the output video. Defaults to FRAME_RATE.
        bitrate (str, optional): The bitrate of the output video. Defaults to BITRATE.
        crf (int, optional): The Constant Rate Factor (CRF) for video compression. Defaults to CRF.
        still_frames (int, optional): The number of still frames to create for each image. Defaults to STILL_FRAMES.
        font_name (str, optional): The name of the font to use for rendering text on the frames. Defaults to FONT_NAME.
        font_size (int, optional): The size of the font to use for rendering text on the frames. Defaults to FONT_SIZE.
    Raises:
        Exception: If FFmpeg pass 1 or pass 2 fails.
    Returns:
        None
    """
    
    run_log: List[FrameDict] = []
    with open(json_file, "r") as f:
        run_log = [json.loads(line) for line in f]

    filtered_run_log = [x for x in run_log if x["failed"] is False]

    source_image = run_log[0]["source_image"]
    frame_images = list(os.path.join(image_dir, x["best_image"]["filename"]) for x in filtered_run_log)

    # Create a temporary working directory
    temp_dir = tempfile.mkdtemp()
    target_files: List[str] = []
    try:
        for frame_num, source_filename in enumerate([source_image] + frame_images):
            frame_filename = f"frame_{str(frame_num).zfill(4)}"
            frame_path = os.path.join(temp_dir, frame_filename) + ".png"

            output_width, output_height = 1920, 1080
            if frame_num == 0:
                # get the dimensions of the second image
                with Image.open(frame_images[0]) as img:
                    output_width, output_height = img.size
                    log.debug(f"source image dimensions: {output_width}x{output_height}")

            log.debug(f"Opening Image: {source_filename}")
            # open the file and write the frame number using PIL
            with Image.open(source_filename) as img:

                # resize frame 0 if necessary
                if frame_num == 0:
                    img = img.resize((output_width, output_height))

                img = render_text(img, str(frame_num), font_name=font_name, font_size=font_size)

                log.debug(f"writing frame {frame_num} from {source_filename} to {frame_path}")
                img.save(frame_path)
            
            target_files.append(frame_path)

        log.debug(f"copied {len(target_files)} files to {temp_dir}")

        # invoke imagemagick to fill in with morph
        frame_ctr = 0

        # create still frames for the first image
        for i in range(still_frames):
            os.symlink(target_files[0], f"{temp_dir}/morph_{str(frame_ctr).zfill(4)}.png")
            frame_ctr += 1

        for ctr in range(1, len(target_files)):
            fname = f"morph_{str(ctr).zfill(4)}"

            log.debug(f"invoking: `convert {target_files[ctr-1]} {target_files[ctr]} -morph {interemediate_frames} {temp_dir}/{fname}_%02d.png`")
            os.system(f"convert {target_files[ctr-1]} {target_files[ctr]} -morph {interemediate_frames} {temp_dir}/{fname}_%02d.png")

            # rename the output files so they just increment the number in frame_ctr
            for i in range(interemediate_frames+2):
                os.rename(f"{temp_dir}/{fname}_{str(i).zfill(2)}.png", f"{temp_dir}/morph_{str(frame_ctr).zfill(4)}.png")
                frame_ctr += 1
            
            # create still frames for the current image
            for i in range(still_frames):
                os.symlink(target_files[ctr], f"{temp_dir}/morph_{str(frame_ctr).zfill(4)}.png")
                frame_ctr += 1

        # use ffmpeg to create the video
        
        cmd1 = f"ffmpeg -r {frame_rate} -i {temp_dir}/morph_%04d.png -c:v libx264 -crf {crf} -b:v {bitrate} -pass 1 -an -f mp4 -y /dev/null"
        log.debug(f"invoking: `{cmd1}`")
        return_code = os.system(cmd1)
        if return_code != 0:
            log.error(f"FFmpeg pass 1 failed with return code {return_code}")
            raise Exception("FFmpeg pass 1 failed")

        cmd2 = f"ffmpeg -r {frame_rate} -i {temp_dir}/morph_%04d.png -c:v libx264 -crf {crf} -b:v {bitrate} -pass 2 -profile:v high -pix_fmt yuv420p -y {output_file}"
        log.debug(f"invoking: `{cmd2}`")
        return_code = os.system(cmd2)
        if return_code != 0:
            log.error(f"FFmpeg pass 2 failed with return code {return_code}")
            raise Exception("FFmpeg pass 2 failed")
        
    finally:
        # Clean up the temporary directory manually
        log.debug(f"cleaning up {temp_dir}")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render video")
    parser.add_argument("--json_file", type=str, help="Path to the JSON file")
    parser.add_argument("--image_dir", type=str, help="Path to the directory containing the frame images")
    parser.add_argument("--output_file", type=str, help="Path to the output video file")
    parser.add_argument("--intermediate_frames", type=int, help="Number of intermediate frames to generate", default=INTERMEDIATE_FRAMES)
    parser.add_argument("--frame_rate", type=int, help="Frame rate of the output video", default=FRAME_RATE)
    parser.add_argument("--still_frames", type=int, help="Number of still frames of each image", default=STILL_FRAMES)

    args = parser.parse_args()
    main(args.json_file, args.image_dir, args.output_file,
         interemediate_frames=args.intermediate_frames, frame_rate=args.frame_rate)
