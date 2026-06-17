from pathlib import Path
from PIL import Image
from pptx import Presentation
from app.schemas import *
from app.pipeline.pptx_builder import build_pptx

def test_pptx_builder_generates_openable_file(tmp_path):
    (tmp_path/'assets').mkdir(); Image.new('RGB',(20,20),'blue').save(tmp_path/'assets'/'image_001.png')
    m=SlideManifest(source=SourceInfo(file_name='sample.png',width_px=400,height_px=225), slide=SlideInfo(), strategy=StrategyInfo(), elements=[ManifestElement(id='shape_001',type='shape',shape='rounded_rect',bbox_px=[20,20,120,60],editable=True,style={'fill':'#FFFFFF','stroke':'#000000'}), ManifestElement(id='image_001',type='image',asset_path='assets/image_001.png',bbox_px=[200,40,80,80],editable=False), ManifestElement(id='text_001',type='text',text='Title',bbox_px=[30,30,100,30],editable=True,style={'font_size':24,'color':'#111111'})], quality=ManifestQuality())
    out=build_pptx(m,tmp_path,tmp_path/'result.pptx')
    assert out.exists() and out.stat().st_size > 0
    Presentation(str(out))
