from transformers import AutoModelForDepthEstimation

def build_network(weights_backbone, **kwargs):
    return AutoModelForDepthEstimation.from_pretrained(weights_backbone, **kwargs)