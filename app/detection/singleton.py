from app.detection.detector import TorchDetector
detector_instance = None

def init_detector():
    global detector_instance
    detector_instance = TorchDetector(use_gpu=False)
    return detector_instance