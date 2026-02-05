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

import os
import sys
import json
import uuid
import logging
from random import randint
from argparse import Namespace, ArgumentParser

import torch
import coloredlogs
from tqdm import tqdm

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import ssim, l1_loss
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.general_utils import safe_state
from scene.gaussian_model import build_scaling_rotation

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
    print("Let's use fused SSIM!")
except ImportError:
    FUSED_SSIM_AVAILABLE = False
    print('Fused SSIM not found, use slow pytorch version.')

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
    print("Let's use sparse Adam!")
except ImportError:
    SPARSE_ADAM_AVAILABLE = False
    print('Sparse Adam not found, use slow pytorch version.')


def training(
    model_params,
    opt_params,
    pipe_params,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    if pipe_params.densification == 'mcmc' and model_params.cap_max == -1:
        print('Please specify the maximum number of Gaussians using --cap_max.')
        sys.exit(1)

    first_iter = 0
    tb_writer = prepare_output_and_logger(model_params)
    gaussians = GaussianModel(model_params.sh_degree, opt_params.optimizer_type)
    scene = Scene(model_params, gaussians)
    gaussians.training_setup(opt_params)

    if model_params.init_prune:
        logging.info('Initial pruning of Gaussians...')
        gaussians.initial_prune()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt_params)

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt_params.optimizer_type == 'sparse_adam' and SPARSE_ADAM_AVAILABLE

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt_params.iterations), desc='Training progress')
    first_iter += 1
    img_num = -1
    img_num_modifier = 1

    for iteration in range(first_iter, opt_params.iterations + 1):
        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if img_num == -1:
                img_num = len(viewpoint_stack)

        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe_params.debug = True

        bg = torch.rand((3), device='cuda') if opt_params.random_background else background

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe_params,
            bg,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            use_trained_exp=model_params.train_test_exp,
        )

        image, viewspace_point_tensor, visibility_filter, radii, gs_w = (
            render_pkg['render'],
            render_pkg['viewspace_points'],
            render_pkg['visibility_filter'],
            render_pkg['radii'],
            render_pkg['gs_w'],
        )

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt_params.lambda_dssim) * Ll1 + opt_params.lambda_dssim * (1.0 - ssim_value)

        # Regularization
        loss = loss + opt_params.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        loss = loss + opt_params.scale_reg * torch.abs(gaussians.get_scaling).mean()

        # anisotropy regularization (PhysGaussian)
        scaling = gaussians.get_scaling
        aniso_ratio = torch.max(scaling, dim=1).values / torch.min(scaling, dim=1).values
        aniso_penalty = torch.relu(aniso_ratio - opt_params.aniso_ratio_thresh).mean()
        loss = loss + opt_params.aniso_reg * aniso_penalty

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                n_gauss = gaussians.get_xyz.shape[0]
                if n_gauss >= 1_000_000:
                    n_gauss_str = f'{n_gauss / 1_000_000:.3f}M'
                elif n_gauss >= 1_000:
                    n_gauss_str = f'{n_gauss / 1000:.1f}k'
                else:
                    n_gauss_str = str(n_gauss)

                progress_bar.set_description(f'loss={ema_loss_for_log:.6f}, #splat={n_gauss_str}')
                progress_bar.update(10)

            if iteration == opt_params.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (
                    pipe_params,
                    background,
                    1.0,
                    SPARSE_ADAM_AVAILABLE,
                    None,
                    model_params.train_test_exp,
                ),
                model_params.train_test_exp,
            )
            if iteration in saving_iterations:
                print(f'\n[ITER {iteration}] Saving Gaussians', end='')
                scene.save(iteration)

            if pipe_params.densification == 'default':
                # Densification (Inria's original)
                if iteration < opt_params.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt_params.densify_from_iter and iteration % opt_params.densification_interval == 0:
                        size_threshold = 20 if iteration > opt_params.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt_params.densify_grad_threshold,
                            None,
                            0.005,
                            scene.cameras_extent,
                            size_threshold,
                            radii,
                        )

                    if iteration % opt_params.opacity_reset_interval == 0 or (
                        model_params.white_background and iteration == opt_params.densify_from_iter
                    ):
                        gaussians.reset_opacity()

            elif pipe_params.densification == 'mcmc':
                # Densification by MCMC
                if (
                    iteration < opt_params.densify_until_iter
                    and iteration > opt_params.densify_from_iter
                    and iteration % opt_params.densification_interval == 0
                ):
                    dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                    gaussians.relocate_gs(dead_mask=dead_mask)
                    gaussians.add_new_gs(cap_max=model_params.cap_max)

            elif pipe_params.densification == 'absgs':
                # Keep track of max weight of each GS for pruning
                gaussians.max_weight[visibility_filter] = torch.max(
                    gaussians.max_weight[visibility_filter],
                    gs_w[visibility_filter],
                )

                if iteration < opt_params.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt_params.densify_from_iter and iteration % opt_params.densification_interval == 0:
                        size_threshold = 20 if iteration > opt_params.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt_params.densify_grad_threshold,
                            opt_params.densify_grad_abs_threshold,
                            0.005,
                            scene.cameras_extent,
                            size_threshold,
                            radii,
                        )

                    if iteration % opt_params.opacity_reduce_interval == 0 and opt_params.use_reduce:
                        gaussians.reduce_opacity()

                    if iteration % opt_params.opacity_reset_interval == 0 or (
                        model_params.white_background and iteration == opt_params.densify_from_iter
                    ):
                        gaussians.reset_opacity()

                if (
                    iteration > opt_params.densify_from_iter
                    and iteration < opt_params.prune_until_iter
                    and opt_params.use_prune_weight
                ):
                    if (
                        iteration % img_num / img_num_modifier == 0
                        and iteration % opt_params.opacity_reset_interval > img_num / img_num_modifier
                    ):
                        prune_mask = (gaussians.max_weight < opt_params.min_weight).squeeze()
                        gaussians.prune_points(prune_mask)
                        gaussians.max_weight *= 0

            # Optimizer step
            if iteration < opt_params.iterations:
                assert gaussians.exposure_optimizer is not None
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)

                if use_sparse_adam:
                    assert isinstance(gaussians.optimizer, SparseGaussianAdam)
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    assert isinstance(gaussians.optimizer, torch.optim.Optimizer)
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                if pipe_params.densification == 'mcmc':
                    # Langevin Monte Carlo noise addition
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)

                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1.0 / (1.0 + torch.exp(-k * (x - x0)))

                    noise = (
                        torch.randn_like(gaussians._xyz)
                        * (op_sigmoid(1.0 - gaussians.get_opacity))
                        * opt_params.noise_lr
                        * xyz_lr
                    )
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)

            if iteration in checkpoint_iterations:
                print(f'\n[ITER {iteration}] Saving Checkpoint', end='')
                torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, f'chkpnt{iteration}.pth'))


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join('./output', unique_str[0:10])

    # Set up output folder
    logging.info(f'Output folder: {args.model_path}')
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, 'cfg_args'), mode='w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        logging.warning('Tensorboard not available: not logging progress')
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    train_test_exp: bool,
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {
                'name': 'train',
                'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            },
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)['render'], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2 :]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2 :]

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            f'{config["name"]}_view_{viewpoint.image_name}/render',
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                f'{config["name"]}_view_{viewpoint.image_name}/ground_truth',
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])

                print(
                    f'\n[ITER {iteration}] {config["name"]}: '
                    f'L1={l1_test:.6f}, PSNR={psnr_test:.2f}, SSIM={ssim_test:.4f}'
                )
                if tb_writer:
                    tb_writer.add_scalar(f'{config["name"]}/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{config["name"]}/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(f'{config["name"]}/loss_viewpoint - ssim', ssim_test, iteration)

        if tb_writer:
            tb_writer.add_histogram('scene/opacity_histogram', scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        torch.cuda.empty_cache()


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


if __name__ == '__main__':
    coloredlogs.install(level='INFO')

    # Set up command line argument parser
    parser = ArgumentParser(description='Training script parameters')
    model_params = ModelParams(parser)
    opt_params = OptimizationParams(parser)
    pipe_params = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[7_000, 15_000, 30_000, 50_000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[7_000, 15_000, 30_000, 50_000])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[7_000, 15_000, 30_000, 50_000])
    parser.add_argument('--start_checkpoint', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        model_params.extract(args),
        opt_params.extract(args),
        pipe_params.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    # All done
    print('\nTraining complete.')
