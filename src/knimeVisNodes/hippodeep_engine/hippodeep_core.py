import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage
import nibabel
from numpy.linalg import inv
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Weights path — torchparams/ sits next to this file
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_WEIGHTS_DIR = os.path.join(_SCRIPT_DIR, "torchparams")
_DEVICE = torch.device("cpu")

# back-compat patch for old torch
try:
    import inspect
    if "align_corners" not in inspect.signature(F.grid_sample).parameters:
        _old = torch.nn.functional.grid_sample
        F.grid_sample = lambda *x, **k: _old(*x)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Constants (unchanged from original)
# ---------------------------------------------------------------------------
_affine64_mni = np.array([
    [-2.85714293,  0.,          0.,         90.],
    [ 0.,          3.42857146,  0.,        -126.],
    [ 0.,          0.,          2.85714293, -72.],
    [ 0.,          0.,          0.,          1.],
])

_bbox_one = np.array([
    [-1,-1,-1,1],[1,-1,-1,1],[-1,1,-1,1],[-1,-1,1,1],
    [1,1,-1,1],[1,-1,1,1],[-1,1,1,1],[1,1,1,1]
])

# ---------------------------------------------------------------------------
# Utility functions — copied verbatim from original hippodeep.py
# ---------------------------------------------------------------------------
def _bbox_world(affine, shape):
    s = shape[0]-1, shape[1]-1, shape[2]-1
    bbox = [[0,0,0],[s[0],0,0],[0,s[1],0],[0,0,s[2]],
            [s[0],s[1],0],[s[0],0,s[2]],[0,s[1],s[2]],[s[0],s[1],s[2]]]
    w = affine @ np.column_stack([bbox, [1]*8]).T
    return w.T

def _mul_homo(g, Mt):
    return g @ Mt[:3,:3].astype(np.float32) + Mt[3,:3].astype(np.float32)

def _indices_unitary(dimensions, dtype):
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,)*N
    res = np.empty((N,)+dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        res[i] = np.linspace(-1, 1, dim, dtype=dtype).reshape(
            shape[:i] + (dim,) + shape[i+1:])
    return res

def _bbox_xyz(shape, affine):
    """World-space corners of a voxel grid."""
    s = shape[0]-1, shape[1]-1, shape[2]-1
    bbox = [[0,0,0],[s[0],0,0],[0,s[1],0],[0,0,s[2]],
            [s[0],s[1],0],[s[0],0,s[2]],[0,s[1],s[2]],[s[0],s[1],s[2]]]
    return _mul_homo(bbox, affine.T)

def _indices_xyz(shape, affine, offset_vox=None):
    """World-space coordinates for every voxel in a grid."""
    if offset_vox is None:
        offset_vox = np.array([0, 0, 0])
    assert len(shape) == 3
    ind = np.indices(shape).astype(np.float32) + \
          offset_vox.reshape(3, 1, 1, 1).astype(np.float32)
    return _mul_homo(np.rollaxis(ind, 0, 4), affine.T)

def _xyz_to_DHW3(xyz, iaffine, srcshape):
    """Normalise world coords to [-1,1] grid_sample convention (DHW, 3)."""
    affine = inv(iaffine)
    ijk3 = _mul_homo(xyz, affine.T)
    ijk3[..., 0] /= srcshape[0] - 1
    ijk3[..., 1] /= srcshape[1] - 1
    ijk3[..., 2] /= srcshape[2] - 1
    ijk3 = ijk3 * 2 - 1
    return np.swapaxes(ijk3, 0, 2)

# ---------------------------------------------------------------------------
# Model definitions — brain mask
# ---------------------------------------------------------------------------
"""a 3D U-Net. Takes the reoriented T1 volume, outputs 4-channel tissue probability maps (brain, CSF, WM, GM). Used to compute eTIV and to guide the affine registration."""
class HeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0a = nn.Conv3d(1, 8, 3, padding=1)
        self.conv0b = nn.Conv3d(8, 8, 3, padding=1)
        self.bn0a   = nn.BatchNorm3d(8)
        self.ma1    = nn.MaxPool3d(2)
        self.conv1a = nn.Conv3d(8, 16, 3, padding=1)
        self.conv1b = nn.Conv3d(16, 24, 3, padding=1)
        self.bn1a   = nn.BatchNorm3d(24)
        self.ma2    = nn.MaxPool3d(2)
        self.conv2a = nn.Conv3d(24, 24, 3, padding=1)
        self.conv2b = nn.Conv3d(24, 32, 3, padding=1)
        self.bn2a   = nn.BatchNorm3d(32)
        self.ma3    = nn.MaxPool3d(2)
        self.conv3a = nn.Conv3d(32, 48, 3, padding=1)
        self.conv3b = nn.Conv3d(48, 48, 3, padding=1)
        self.bn3a   = nn.BatchNorm3d(48)
        self.conv2u = nn.Conv3d(48, 24, 3, padding=1)
        self.conv2v = nn.Conv3d(24+32, 24, 3, padding=1)
        self.bn2u   = nn.BatchNorm3d(24)
        self.conv1u = nn.Conv3d(24, 24, 3, padding=1)
        self.conv1v = nn.Conv3d(24+24, 24, 3, padding=1)
        self.bn1u   = nn.BatchNorm3d(24)
        self.conv0u = nn.Conv3d(24, 16, 3, padding=1)
        self.conv0v = nn.Conv3d(16+8, 8, 3, padding=1)
        self.bn0u   = nn.BatchNorm3d(8)
        self.conv1x = nn.Conv3d(8, 4, 1, padding=0)

    def forward(self, x):
        x = F.elu(self.conv0a(x))
        self.li0 = x = F.elu(self.bn0a(self.conv0b(x)))
        x = self.ma1(x)
        x = F.elu(self.conv1a(x))
        self.li1 = x = F.elu(self.bn1a(self.conv1b(x)))
        x = self.ma2(x)
        x = F.elu(self.conv2a(x))
        self.li2 = x = F.elu(self.bn2a(self.conv2b(x)))
        x = self.ma3(x)
        x = F.elu(self.conv3a(x))
        x = F.elu(self.bn3a(self.conv3b(x)))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.elu(self.conv2u(x))
        x = torch.cat([x, self.li2], 1)
        x = F.elu(self.bn2u(self.conv2v(x)))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.elu(self.conv1u(x))
        x = torch.cat([x, self.li1], 1)
        x = F.elu(self.bn1u(self.conv1v(x)))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.elu(self.conv0u(x))
        x = torch.cat([x, self.li0], 1)
        x = F.elu(self.bn0u(self.conv0v(x)))
        return torch.sigmoid(self.conv1x(x))

"""
an affine registration network. Takes 2 channels from HeadModel output, predicts a 3*4 affine matrix that aligns the brain to MNI space. 
This is a learned rigid/affine registration, not iterative optimisation.
"""

class ModelAff(nn.Module):
    def __init__(self):
        super().__init__()
        self.convaff1 = nn.Conv3d(2, 16, 3, padding=1)
        self.maaff1   = nn.MaxPool3d(2)
        self.convaff2 = nn.Conv3d(16, 16, 3, padding=1)
        self.bnaff2   = nn.LayerNorm([32, 32, 32])
        self.maaff2   = nn.MaxPool3d(2)
        self.convaff3 = nn.Conv3d(16, 32, 3, padding=1)
        self.bnaff3   = nn.LayerNorm([16, 16, 16])
        self.maaff3   = nn.MaxPool3d(2)
        self.convaff4 = nn.Conv3d(32, 64, 3, padding=1)
        self.maaff4   = nn.MaxPool3d(2)
        self.bnaff4   = nn.LayerNorm([8, 8, 8])
        self.convaff5 = nn.Conv3d(64, 128, 1, padding=0)
        self.convaff6 = nn.Conv3d(128, 12, 4, padding=0)

        gsx, gsy, gsz = 64, 64, 64
        gx = np.linspace(-1, 1, gsx)
        gy = np.linspace(-1, 1, gsy)
        gz = np.linspace(-1, 1, gsz)
        grid = np.meshgrid(gx, gy, gz)
        grid = np.stack([grid[2], grid[1], grid[0], np.ones_like(grid[0])], axis=3)
        netgrid = np.swapaxes(grid, 0, 1)[...,[2,1,0,3]]
        self.register_buffer('grid', torch.tensor(netgrid.astype("float32"), requires_grad=False))
        self.register_buffer('diagA', torch.eye(4, dtype=torch.float32))

    def forward(self, outc1):
        x = outc1
        x = F.relu(self.convaff1(x));  x = self.maaff1(x)
        x = F.relu(self.bnaff2(self.convaff2(x))); x = self.maaff2(x)
        x = F.relu(self.bnaff3(self.convaff3(x))); x = self.maaff3(x)
        x = F.relu(self.bnaff4(self.convaff4(x))); x = self.maaff4(x)
        x = F.relu(self.convaff5(x))
        x = self.convaff6(x)
        x = x.view(-1, 3, 4)
        x = torch.cat([x, x[:,0:1] * 0], dim=1)
        self.tA = torch.transpose(x + self.diagA, 1, 2)
        wgrid = self.grid @ self.tA[:,None,None]
        return F.grid_sample(outc1, wgrid[...,[2,1,0]], align_corners=True), self.tA

"""
HippoModel — the hippocampus segmentation network. A custom encoder-decoder with skip connections and a blur-guided
attention mechanism. Runs separately on left and right hippocampus crops 
(extracted from the MNI-aligned ROI with mirroring for the left side).
"""
class HippoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0a_0  = nn.Conv3d(1, 16, (1,1,3), padding=0)
        self.conv0a_1  = nn.Conv3d(16, 16, (1,3,1), padding=0)
        self.conv0a    = nn.Conv3d(16, 16, (3,1,1), padding=0)
        self.convf1    = nn.Conv3d(16, 48, (3,3,3), padding=0)
        self.maxpool1  = nn.MaxPool3d(2)
        self.bn1       = nn.BatchNorm3d(48, momentum=1); self.bn1.training = False
        self.convout0  = nn.Conv3d(48, 48, (3,3,3), padding=1)
        self.convout1  = nn.Conv3d(48, 48, (3,3,3), padding=1)
        self.maxpool2  = nn.MaxPool3d(2)
        self.bn2       = nn.BatchNorm3d(48, momentum=1); self.bn2.training = False
        self.convout2p = nn.Conv3d(48, 48, (3,3,3), padding=1)
        self.convout2  = nn.Conv3d(48, 48, (3,3,3), padding=1)
        self.convlx3   = nn.Conv3d(48, 48, (3,3,3), padding=1)
        self.convlx5   = nn.Conv3d(48, 48, (3,3,3), padding=1)
        self.convlx7   = nn.Conv3d(48, 16, (3,3,3), padding=1)
        self.convlx8   = nn.Conv3d(16, 1, 1, padding=0)
        self.blur      = nn.Conv3d(1, 1, 7, padding=3)
        self.conv_extract = nn.Conv3d(48, 47, 3, padding=1)
        self.convmix   = nn.Conv3d(48, 16, 3, padding=1)
        self.convout1x = nn.Conv3d(16, 1, 1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv0a_0(x))
        x = F.relu(self.conv0a_1(x))
        x = F.relu(self.conv0a(x))
        self.out_conv_f1  = x = F.relu(self.convf1(x))
        self.out_maxpool1 = x = self.maxpool1(x)
        x = self.bn1(x)
        x = F.relu(self.convout0(x))
        x = self.convout1(x)
        x = F.relu(x + self.out_maxpool1)
        self.out_maxpool2 = x = self.maxpool2(x)
        x = self.bn2(x)
        x = F.relu(self.convout2p(x))
        x = self.convout2(x)
        x = F.relu(x + self.out_maxpool2)

        # NOTE: self.lx2 is stored as an attribute but intentionally never used
        # again in this forward pass. x must NOT be overwritten here — convlx3
        # must operate on the smaller post-pooling x, not the interpolated one.
        # Writing `x = F.interpolate(...)` instead of `self.lx2 = ...` causes
        # x to be doubled one step too early, making x * out_conv_f1 fail with
        # a "size of tensor a (120) must match tensor b (60)" error.
        self.lx2 = F.interpolate(x, scale_factor=2, mode="nearest")

        x = F.relu(self.convlx3(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.convlx5(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.convlx7(x))
        self.out_output1 = x = torch.sigmoid(self.convlx8(x))
        x = torch.sigmoid(self.blur(x))
        x = x * self.out_conv_f1
        x = F.leaky_relu(self.conv_extract(x))
        x = torch.cat([self.out_output1, x], dim=1)
        x = F.relu(self.convmix(x))
        return torch.sigmoid(self.convout1x(x))


# ---------------------------------------------------------------------------
# One-time model loading at import time
# ---------------------------------------------------------------------------
_net = HeadModel().to(_DEVICE)
_net.load_state_dict(torch.load(
    os.path.join(_WEIGHTS_DIR, "params_head_00075_00000.pt"),
    map_location=_DEVICE))
_net.eval()

_netAff = ModelAff().to(_DEVICE)
_netAff.load_state_dict(torch.load(
    os.path.join(_WEIGHTS_DIR, "paramsaffineta_00079_00000.pt"),
    map_location=_DEVICE), strict=False)
_netAff.eval()

_hipponet = HippoModel().to(_DEVICE)
_hipponet.load_state_dict(torch.load(
    os.path.join(_WEIGHTS_DIR, "hippodeep.pt"),
    map_location=_DEVICE))
_hipponet.eval()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class HippoResult:
    etiv_mm3:     float
    hippo_l_mm3:  float
    hippo_r_mm3:  float
    mask_l:       object   # np.ndarray, native image space, uint8 0-255
    mask_r:       object   # np.ndarray, native image space, uint8 0-255
    affine:       object   # np.ndarray, native NIfTI affine
    coronal_axis: int      # voxel axis aligned with A-P direction


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def segment_hippocampus(img_path: str) -> HippoResult:
    """
    Run the full HippoDeep pipeline on one T1 NIfTI image.
    Returns volumes (mm³) and 3D mask arrays in native image space.
    No files written, no subprocess.
    """
    img = nibabel.load(img_path)

    #check img object of the class 
    #qform (Quaternion form): Describes orientation relative to the scanner coordinates.
    #sform (Standard form): Describes orientation mapped to a standard anatomical space (like MNI template space).

    if isinstance(img, nibabel.nifti1.Nifti1Image):
        img._affine = img.get_qform()
    if isinstance(img, nibabel.Nifti1Image):
        # qform -> scanner coordinates 
        # sform -> standard coordinates
        # affine -> transformation matrix perform conversion
        # tolerating small differences / comparing s_form and q_form if significant -> prioritizes qform and explicitly overrides internal _affine attribute with get_qform().
        if not np.allclose(img.get_sform(), img.get_qform(), atol=1e-4): 
            img._affine = img.get_qform()

    #convert image object into NumPy array 
    d = img.get_fdata(caching="unchanged", dtype=np.float32) # unchanged -> not to RAM for NumPy array
    while d.ndim > 3: # 4 or 5D into 3D 
        d = d.mean(-1) # (128, 128, 64, 20) -> averages across 20 timpepoints -> (128, 128, 64) 3D
    d = (d - d.mean()) / d.std() #mean 0 and std 1 -> z-score normlaization 

    # Coronal axis derived from affine (world Y = anterior-posterior)
    coronal_axis = int(np.argmax(np.abs(img.affine[:3, :3][1, :]))) # 3*3 rotation and scaling matrix + Y-axis -> anterior-posterior or front-back direction 

    # Orientation transform to LAS
    # o1: Reads the image's current spatial orientation matrix from img.affine (e.g., RAS, LPI, etc.).
    # o2: Defines the target orientation matrix for LAS:
    # [0., -1.]: Axis 0 points towards Left (negative direction).
    # [1.,  1.]: Axis 1 points towards Anterior (positive direction).
    # [2.,  1.]: Axis 2 points towards Superior (positive direction).
    # trn: Calculates the transformation required to reorient array axes from the current orientation (o1) to the target LAS orientation (o2).

    o1       = nibabel.orientations.io_orientation(img.affine)
    o2       = np.array([[0.,-1.],[1.,1.],[2.,1.]])
    trn      = nibabel.orientations.ornt_transform(o1, o2)
    trn_back = nibabel.orientations.ornt_transform(o2, o1)
    revaff1  = nibabel.orientations.inv_ornt_aff(trn,      (1,1,1))
    revaff1i = nibabel.orientations.inv_ornt_aff(trn_back, (1,1,1))

    aff_orig64 = np.linalg.lstsq(
        _bbox_world(np.identity(4), (64,64,64)),
        _bbox_world(img.affine, img.shape[:3]), rcond=None)[0].T
    voxscale_native64 = np.abs(np.linalg.det(aff_orig64))

    revaff64i  = nibabel.orientations.inv_ornt_aff(trn_back, (64,64,64))

    wgridt = (_netAff.grid @ torch.tensor(
        revaff1i, device=_DEVICE, dtype=torch.float32))[None,...,[2,1,0]]
    d_orr = F.grid_sample(
        torch.as_tensor(d, dtype=torch.float32, device=_DEVICE)[None,None],
        wgridt, align_corners=True)

    # HeadModel — brain/tissue priors
    with torch.no_grad():
        out1t = _net(d_orr)
    out1 = np.asarray(out1t.cpu())

    # eTIV from brain mask resampled to native space
    gsx, gsy, gsz = img.shape[:3]
    sgrid = np.rollaxis(_indices_unitary((gsx,gsy,gsz), dtype=np.float16), 0, 4)
    wgridt_nat = torch.as_tensor(
        _mul_homo(sgrid, inv(revaff1i))[None,...,[2,1,0]],
        device=_DEVICE, dtype=torch.float32)
    del sgrid
    brain_out = out1[0,0].astype("float32")
    dnat = np.asarray(F.grid_sample(
        torch.as_tensor(brain_out, dtype=torch.float32, device=_DEVICE)[None,None],
        wgridt_nat, align_corners=True).cpu())[0,0]
    etiv_mm3 = float((dnat > .5).sum() * np.abs(np.linalg.det(img.affine)))
    del dnat
    brainmask_cc = torch.tensor(brain_out)

    # ModelAff — MNI affine registration
    with torch.no_grad():
        wc1, tA = _netAff(out1t[:,[1,3]] * brainmask_cc)
    wnat = np.linalg.lstsq(
        _bbox_world(img.affine, img.shape[:3]), _bbox_one @ revaff1, rcond=None)[0]
    wmni = np.linalg.lstsq(
        _bbox_world(_affine64_mni, (64,64,64)), _bbox_one, rcond=None)[0]
    M = (wnat @ inv(np.asarray(tA[0].cpu())) @ inv(wmni)).T

    # HippoModel — segment in MNI-cropped ROI
    imgcroproi_affine = np.array([
        [-1., 0., 0., 54.],
        [ 0., 1., 0., -59.],
        [ 0., 0., 1., -45.],
        [ 0., 0., 0.,  1.]])
    imgcroproi_shape = (107, 72, 68)

    gsx2, gsy2, gsz2 = imgcroproi_shape
    sgrid2 = np.rollaxis(_indices_unitary((gsx2,gsy2,gsz2), dtype=np.float32), 0, 4)
    bboxnat = _bbox_world(imgcroproi_affine, imgcroproi_shape) @ inv(M.T) @ wnat
    matzoom = np.linalg.lstsq(_bbox_one, bboxnat, rcond=None)[0]
    wgridt2 = torch.tensor(
        _mul_homo(sgrid2, (matzoom @ revaff1i))[None,...,[2,1,0]],
        device=_DEVICE, dtype=torch.float32)
    dout = F.grid_sample(
        torch.as_tensor(d, dtype=torch.float32, device=_DEVICE)[None,None],
        wgridt2, align_corners=True)
    d_in = np.asarray(dout[0,0].cpu())
    d_in -= d_in.mean(); d_in /= d_in.std()

    with torch.no_grad():
        hippoR = _hipponet(torch.as_tensor(d_in[None, None,  6: 54:+1, :, 2:-2].copy()))
        hippoL = _hipponet(torch.as_tensor(d_in[None, None, -7:-55:-1, :, 2:-2].copy()))

    hippoRL = np.vstack([np.asarray(hippoR.cpu()), np.asarray(hippoL.cpu())])
    hippoRL = np.clip(((hippoRL - .5) * 2 + .5), 0, 1) * (hippoRL > .5)

    output_arr = np.zeros((2, 107, 72, 68), np.float32)
    output_arr[0, -7:-55:-1, :, 2:-2][2:-2, 2:-2, 2:-2] = np.clip(hippoRL[1]*255, 0, 255)
    output_arr[1,  6: 54:+1, :, 2:-2][2:-2, 2:-2, 2:-2] = np.clip(hippoRL[0]*255, 0, 255)

    boxvols = hippoRL[[1,0]].reshape(2,-1).sum(1) * \
              np.abs(np.linalg.det(imgcroproi_affine @ inv(M)))
    hippo_l_mm3 = float(boxvols[0])
    hippo_r_mm3 = float(boxvols[1])

    # Resample masks back to native space — using the three helper functions
    def _resample_to_native(channel):
        # Find bounding box of the ROI crop in native image voxel space
        pts     = _bbox_xyz(imgcroproi_shape, imgcroproi_affine)
        pts     = _mul_homo(pts, inv(M).T)
        pts_ijk = _mul_homo(pts, inv(img.affine).T)
        for i in range(3):
            np.clip(pts_ijk[:, i], 0, img.shape[i], out=pts_ijk[:, i])
        pmin   = np.floor(np.min(pts_ijk, 0)).astype(int)
        pwidth = np.ceil( np.max(pts_ijk, 0)).astype(int) - pmin

        # Build world-coordinate grid for each native voxel in the bounding box,
        # transform to MNI crop space, normalise to [-1,1] for grid_sample
        widx = _indices_xyz(pwidth, img.affine, offset_vox=pmin)
        widx = _mul_homo(widx, M.T)
        DHW3 = _xyz_to_DHW3(widx, imgcroproi_affine, imgcroproi_shape)

        # Sample the probability map onto the native bounding box
        d_t  = torch.tensor(output_arr[channel].T, dtype=torch.float32)
        outD = F.grid_sample(
            d_t[None, None],
            torch.tensor(DHW3[None], dtype=torch.float32),
            align_corners=True)
        dnat = np.asarray(outD[0, 0].permute(2, 1, 0))
        dnat[dnat < 32] = 0

        # Place the bounding-box result into a full native-space volume
        wdata = np.zeros(img.shape[:3], np.uint8)
        wdata[pmin[0]:pmin[0]+pwidth[0],
              pmin[1]:pmin[1]+pwidth[1],
              pmin[2]:pmin[2]+pwidth[2]] = dnat.astype(np.uint8)
        return wdata

    mask_l_arr = _resample_to_native(0)
    mask_r_arr = _resample_to_native(1)

    return HippoResult(
        etiv_mm3    = etiv_mm3,
        hippo_l_mm3 = hippo_l_mm3,
        hippo_r_mm3 = hippo_r_mm3,
        mask_l      = mask_l_arr,
        mask_r      = mask_r_arr,
        affine      = img.affine,
        coronal_axis= coronal_axis,
    )