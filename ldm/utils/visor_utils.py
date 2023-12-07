import os
import cv2
import numpy as np
import os
from numpy import asarray

# note that shapes are width,height not rows,columns
def mask_to_img(masks_info, object_keys=None, input_resolution=(1920,1080), output_resolution=(1920,1080)):
    non_empty_objects = []
    img = np.zeros([input_resolution[1],input_resolution[0],3],dtype=np.uint8)
    index = 1
    entities = masks_info
    i = 1
    for entity in entities:
        object_annotations = entity["segments"]
        polygons = []
        polygons.append(object_annotations)
        non_empty_objects.append(entity["name"])
        ps = []
        for polygon in polygons:
            for poly in polygon:
                if poly == []:
                    poly = [[0.0, 0.0]]
                ps.append(np.array(poly, dtype=np.int32))
        if object_keys:
            obj_name = entity['name'].replace('/',' ')
            if (obj_name in object_keys.keys()):
                if isinstance(object_keys[obj_name],tuple):
                    color = object_keys[obj_name]                
                else:
                    color = (object_keys[obj_name], object_keys[obj_name], object_keys[obj_name])
                cv2.fillPoly(img, ps, color)
        else:
            cv2.fillPoly(img, ps, (i, i, i))
        i += 1
    image_data = asarray(img)
    if (input_resolution != output_resolution):
        out_image = cv2.resize(image_data, (output_resolution[0],
                                    output_resolution[1]),
                             interpolation=cv2.INTER_NEAREST)
        out_image = (np.array(out_image)).astype('uint8')
    else:
        out_image = image_data
    return out_image

