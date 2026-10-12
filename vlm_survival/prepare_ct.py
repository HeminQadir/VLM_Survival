"""NIfTI -> windowed fixed-size array. Baseline preprocessing, not tumor localization."""
import argparse
import json
from pathlib import Path
import numpy as np


def prepare(source, output, shape, low=-150, high=250):
    import nibabel as nib
    from scipy.ndimage import zoom
    if high <= low or any(s < 8 for s in shape):
        raise ValueError('Invalid CT window or target dimensions.')
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    if out.suffix != '.npy':
        raise ValueError('Output must end in .npy.')
    image = nib.as_closest_canonical(nib.load(source))
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError('Expected one finite 3D CT series in Hounsfield units.')
    # NIfTI canonical axes are x,y,z; transpose for Conv3d depth,height,width.
    volume = volume.transpose(2, 1, 0)
    source_shape = volume.shape
    volume = (np.clip(volume, low, high)-low)/(high-low)
    resized = zoom(volume, np.asarray(shape)/np.asarray(volume.shape), order=1, prefilter=False)
    if resized.shape != tuple(shape):
        raise ValueError(f'Unexpected resampling shape {resized.shape}')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, resized.astype('float32'))
    out.with_suffix('.json').write_text(json.dumps({
        'canonical_shape_xyz': list(image.shape), 'source_shape_dhw': list(source_shape),
        'source_spacing_xyz_mm': [float(x) for x in image.header.get_zooms()[:3]],
        'canonical_affine': image.affine.tolist(), 'target_shape_dhw': shape,
        'window_hu': [low, high],
        'warning': 'Whole-volume resize changes physical scale and can erase small lesions. No liver cropping, phase registration or lesion segmentation is performed.'}, indent=2))
    print(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--shape', type=int, nargs=3, default=[64, 96, 96])
    p.add_argument('--low', type=float, default=-150)
    p.add_argument('--high', type=float, default=250)
    a = p.parse_args()
    prepare(a.input, a.output, a.shape, a.low, a.high)
