import pyiqa
import torch
import logging

class NIMAEvaluator:
    def __init__(self) -> None:    
        self.log = logging.getLogger(self.__class__.__name__)

        # Check if MPS backend is available
        device = None
        if torch.backends.mps.is_available():
            # device = torch.device("mps")
            device = torch.device("cpu")
            self.log.warn("Using CPU backend instead of MPS due to implementation issues")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            self.log.info("Using CUDA")
        else:
            device = torch.device("cpu")
            self.log.info("MPS and CUDA backends not available, using CPU")

        self.device = device
        self.metrics = {}

    def evaluate(self, image_path: str) -> float:
        # create metric with default setting
        if "nima" not in self.metrics:
            self.metrics["nima"] = pyiqa.create_metric('nima', device=self.device)

        iqa_metric = self.metrics["nima"]

        # check if lower better or higher better
        # print(iqa_metric.lower_better)

        # img path as inputs.
        score_fr = iqa_metric(image_path)
        # print(score_fr)
        return float(score_fr[0][0])

if __name__ == "__main__":
    evaluator = NIMAEvaluator()
    score = evaluator.evaluate("test_image.jpg")
    print(f"The aesthetic quality of the image is: {score}")