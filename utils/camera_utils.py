#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import multiprocessing

import cv2
import numpy as np
import torch
import joblib
import torch.nn as nn

from utils.graphics_utils import fov2focal, getWorld2View2, getProjectionMatrix

WARNED = False


class Camera(nn.Module):
    def __init__(
        self,
        resolution: tuple[int, int],
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        depth_params,
        image,
        invdepthmap,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device='cuda',
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f'[Warning] Custom device {data_device} failed, fallback to default cuda device')
            self.data_device = torch.device('cuda')

        resized_image_rgb = cv2.resize(image, resolution, interpolation=cv2.INTER_AREA)
        resized_image_rgb = torch.from_numpy(resized_image_rgb).to(self.data_device)
        resized_image_rgb = (resized_image_rgb / 255.0).permute(2, 0, 1)

        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].clone()
        else:
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...])

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., : self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2 :] = 0

        self.original_image = gt_image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if (
                    depth_params['scale'] < 0.2 * depth_params['med_scale']
                    or depth_params['scale'] > 5 * depth_params['med_scale']
                ):
                    self.depth_reliable = False
                    self.depth_mask *= 0

                if depth_params['scale'] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params['scale'] + depth_params['offset']

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        self.world_view_transform = nn.Parameter(world_view_transform, requires_grad=False)
        self.projection_matrix = nn.Parameter(projection_matrix, requires_grad=False)
        self.full_proj_transform = nn.Parameter(full_proj_transform, requires_grad=False)
        self.camera_center = nn.Parameter(camera_center, requires_grad=False)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic: bool, is_test_dataset: bool):
    image = cv2.imread(cam_info.image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Failed to load image at {cam_info.image_path}')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if cam_info.depth_path != '':
        try:
            invdepthmap = cv2.imread(cam_info.depth_path, cv2.IMREAD_UNCHANGED)
            if invdepthmap is None:
                raise FileNotFoundError(f'Failed to load image at {cam_info.depth_path}')

            if is_nerf_synthetic:
                invdepthmap = invdepthmap.astype(np.float32) / 512
            else:
                invdepthmap = invdepthmap.astype(np.float32) / float(2**16)

        except FileNotFoundError as e:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise e

        except IOError as e:
            print(
                f"Error: Unable to open the image file '{cam_info.depth_path}'. "
                'It may be corrupted or an unsupported format.'
            )
            raise e

        except Exception as e:
            print(f'An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}')
            raise e
    else:
        invdepthmap = None

    orig_h, orig_w = image.shape[:2]
    if args.resolution in [1, 2, 4, 8]:
        resolution = (
            round(orig_w / (resolution_scale * args.resolution)),
            round(orig_h / (resolution_scale * args.resolution)),
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        '[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n '
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(
        resolution,
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        depth_params=cam_info.depth_params,
        image=image,
        invdepthmap=invdepthmap,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        train_test_exp=args.train_test_exp,
        is_test_dataset=is_test_dataset,
        is_test_view=cam_info.is_test,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic: bool, is_test_dataset: bool):
    camera_list = []

    n_jobs = multiprocessing.cpu_count() // 2
    chunksize = n_jobs * 8
    n_chunks = (len(cam_infos) + chunksize - 1) // chunksize

    for i in range(n_chunks):
        start_idx = i * chunksize
        end_idx = min((i + 1) * chunksize, len(cam_infos))
        chunk_cameras = joblib.Parallel(n_jobs=n_jobs, backend='threading')(
            joblib.delayed(loadCam)(
                args,
                id + start_idx,
                c,
                resolution_scale,
                is_nerf_synthetic,
                is_test_dataset,
            )
            for id, c in enumerate(cam_infos[start_idx:end_idx], start=start_idx)
        )
        camera_list.extend(chunk_cameras)

    camera_list = [c.to('cuda') for c in camera_list]
    return camera_list


def camera_to_JSON(id: int, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
