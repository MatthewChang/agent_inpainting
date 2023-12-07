import torch
def predict_batch(predictor, images):
    """
    Args:
        predictor: Default predictor from detectron2
        images (np.ndarray): an images of shape (B, H, W, C) (in RGB order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        inputs = []
        for original_image in images:
            # Apply pre-processing to image.
            if predictor.input_format == "BGR":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append({"image": image, "height": height, "width": width})
        predictions = predictor.model(inputs)
        return predictions

