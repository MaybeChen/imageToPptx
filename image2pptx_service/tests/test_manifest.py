from app.schemas import SlideManifest, SourceInfo, SlideInfo, StrategyInfo, ManifestElement, ManifestQuality

def test_manifest_schema_requires_core_fields():
    m=SlideManifest(source=SourceInfo(file_name='x.png', width_px=100, height_px=100), slide=SlideInfo(), strategy=StrategyInfo(), elements=[ManifestElement(id='text_001', type='text', text='Hello', bbox_px=[1,2,3,4], editable=True)], quality=ManifestQuality())
    d=m.model_dump()
    assert {'version','source','slide','elements'} <= set(d)
    for e in d['elements']:
        assert {'id','type','bbox_px','editable'} <= set(e)
        assert len(e['bbox_px']) == 4
        assert all(isinstance(n, (int,float)) for n in e['bbox_px'])
