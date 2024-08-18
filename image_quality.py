from typing import Tuple, List, Any, Dict
import cv2
import os
import argparse
from nima_aesthetics import NIMAEvaluator
import logging

DEFAULT_THRESHOLD = 60.0

class QualityEvaluator:
    def __init__(self, evaluator: NIMAEvaluator=None, threshold: float=DEFAULT_THRESHOLD) -> None:
        self.log = logging.getLogger(self.__class__.__name__)
        self.evaluator = evaluator or NIMAEvaluator()
        self.threshold = threshold or DEFAULT_THRESHOLD
                
    def is_image_blurry(self, image_path: str, variance: int=None) -> Tuple[bool, float]:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply Laplacian operator
        laplacian = cv2.Laplacian(image, cv2.CV_64F)

        # Calculate the variance of the Laplacian
        var = laplacian.var()

        thresh = variance or self.threshold
        # Check if the variance is below the threshold
        if var < thresh:
            return (True, float(var))
        else:
            return (False, float(var))

    def process_image(self, image_path: str, variance: int=None) -> Dict[str, Any]:
        score = self.evaluator.evaluate(image_path)
        blurry, variance = self.is_image_blurry(image_path, variance)
        rv = {"blurry": { "value": blurry, "variance": variance }, "NIMA_score": score}
        self.log.info(f"Image: {image_path}, Blurry: {blurry}, Variance: {variance}, NIMA Score: {score}")
        return rv

    def process_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:

        # Iterate over the image paths
        for image_path in image_paths:
            # Check if the path is a file
            if os.path.isfile(image_path):
                self.process_image(image_path)
            # Check if the path is a directory
            elif os.path.isdir(image_path):
                # Iterate over the image files in the directory
                for file_name in os.listdir(image_path):
                    # Check if the file is an image
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        self.process_image(os.path.join(image_path, file_name))
            else:
                self.log.warn(f"Invalid path: {image_path}")

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Check if an image is blurry.")

    # Add the image path argument
    parser.add_argument("image_path", nargs="+", type=str, help="Path to the image files or directories")

    # Add the threshold argument
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Blurry threshold")

    # Parse the command line arguments
    args = parser.parse_args()

    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    
    qe = QualityEvaluator(threshold=args.threshold)
    qe.process_images(args.image_path)
    # process_images(args.image_path, args.threshold)
