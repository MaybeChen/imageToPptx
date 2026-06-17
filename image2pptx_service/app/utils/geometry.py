def iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1=max(ax,bx); y1=max(ay,by); x2=min(ax+aw,bx+bw); y2=min(ay+ah,by+bh)
    inter=max(0,x2-x1)*max(0,y2-y1)
    union=aw*ah + bw*bh - inter
    return inter/union if union else 0.0

def px_to_inches(bbox, source_w, source_h, slide_w, slide_h):
    x,y,w,h = bbox
    return (x/source_w*slide_w, y/source_h*slide_h, w/source_w*slide_w, h/source_h*slide_h)
