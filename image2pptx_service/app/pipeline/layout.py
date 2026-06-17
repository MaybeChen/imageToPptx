from app.utils.geometry import iou
LAYER={'background':0,'shape':1,'image':2,'icon':2,'chart':2,'line':3,'arrow':3,'text':4}

def combine_layout(ocr_items, segments):
    filtered=[]
    for seg in segments:
        if any(iou(seg.bbox_px, o.bbox_px) > 0.45 for o in ocr_items):
            continue
        filtered.append(seg)
    return sorted(filtered, key=lambda s: (LAYER.get(s.type, 9), s.bbox_px[1], s.bbox_px[0]))
