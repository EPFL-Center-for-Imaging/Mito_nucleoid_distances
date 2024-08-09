import pathlib
import random

import napari
import napari_animation
import numpy as np
import scipy
import tifffile

import main

if __name__ == "__main__":
    pixel_size_um = np.array([0.2, 0.056, 0.056])

    img_path = pathlib.Path(
        "../Image IND analysis test/240531_dsDNA-TOM20-siMFF_004_decon_crop2/240531_dsDNA-TOM20-siMFF_004_decon_crop2.tif"
    )
    # img_path = pathlib.Path(
    #     "../Image IND analysis test/240531_dsDNA-TOM20-siINF2_003_decon_crop5/240531_dsDNA-TOM20-siINF2_003_decon_crop5.tif"
    # )
    img = tifffile.imread(img_path)

    labels_path = (
        img_path.parent / f"nellie_output/C2-{img_path.stem}-ch0-im_skel.ome.tif"
    )
    labels_img = tifffile.imread(labels_path)
    binary_skeleton = main.label_img_to_binary_skeleton(labels_img, min_length_px=20)

    splines_um = main.skeleton2splines(
        binary_skeleton, pixel_size_um, knots2data_ratio=5, extension_um=0.3
    )
    random.shuffle(splines_um)
    tracks_um = []
    max_n_t = max([spline_um.M * 10 for spline_um in splines_um])
    normal_plane_corners_um = [[] for _ in range(max_n_t)]
    normal_planes_um = []
    points = []

    for spline_id, spline_um in enumerate(splines_um):
        t = np.linspace(0, spline_um.M - 1, spline_um.M * 10)
        track_um = np.zeros((len(t), 5))
        track_um[:, 0] = spline_id
        track_um[:, 1] = np.arange(len(t))
        track_um[:, 2:] = spline_um.eval(t)
        tracks_um.append(track_um)

        spline_normal_planes_um = main.compute_normal_planes(
            spline_um, t, pixel_size_um, half_window_size_um=0.5
        )
        normal_plane_um = np.zeros((len(t), 4, 4))
        normal_plane_frames = np.arange(len(t)) + max_n_t + 5
        normal_plane_um[:, :, 0] = np.tile(normal_plane_frames, (4, 1)).T
        normal_plane_um[:, 0, 1:] = spline_normal_planes_um[:, 0, 0]
        normal_plane_um[:, 1, 1:] = spline_normal_planes_um[:, 0, -1]
        normal_plane_um[:, 2, 1:] = spline_normal_planes_um[:, -1, -1]
        normal_plane_um[:, 3, 1:] = spline_normal_planes_um[:, -1, 0]
        normal_planes_um.append(normal_plane_um)

        normal_imgs_nucleoid = main.extract_normal_plane_imgs(
            spline_normal_planes_um, img[:, 0], pixel_size_um
        )
        intensity_nucleoid = np.nanmean(normal_imgs_nucleoid, axis=(1, 2))
        peak_indices, _ = scipy.signal.find_peaks(intensity_nucleoid, prominence=320)
        max_normal_plane_frames = 2 * max_n_t + 5
        for peak_index in peak_indices:
            point_frames = np.arange(
                normal_plane_frames[peak_index], max_normal_plane_frames + 1
            )
            spline_point = np.zeros((len(point_frames), 4))
            spline_point[:, 0] = point_frames
            spline_point[:, 1:] = track_um[:, 2:][peak_index]
            points.append(spline_point)

    tracks_um = np.concatenate(tracks_um, axis=0)
    normal_planes_um = np.concatenate(normal_planes_um, axis=0)
    points = np.concatenate(points, axis=0)

    viewer = napari.Viewer(ndisplay=3)

    image_layers = viewer.add_image(
        img,
        channel_axis=1,
        gamma=[2, 1],
        contrast_limits=[
            [np.min(img[:, 0]), np.max(img[:, 0]) * 0.5],
            [np.min(img[:, 1]), np.max(img[:, 1])],
        ],
        scale=pixel_size_um,
        depiction="plane",
        blending="additive",
    )
    image_layer_nucleoid = image_layers[0]
    image_layer_mito = image_layers[1]

    binary_layer = viewer.add_image(
        binary_skeleton,
        scale=pixel_size_um,
        blending="translucent",
    )
    binary_layer.visible = False

    splines_layer = viewer.add_tracks(
        tracks_um,
        tail_width=10,
        tail_length=1000,
        blending="translucent",
    )
    splines_layer.visible = False

    normal_plane_layer = viewer.add_shapes(
        normal_planes_um, shape_type="rectangle", edge_width=0
    )
    normal_plane_layer.visible = False

    points_layer = viewer.add_points(points, size=0.4, edge_width=0, opacity=0.5)
    points_layer.visible = False

    animation = napari_animation.Animation(viewer)
    viewer.update_console({"animation": animation})

    viewer.camera.angles = (-18.23797054423494, 41.97404742075617, 141.96173085742896)
    viewer.camera.zoom *= 1.4

    def replace_binary_data():
        z_cutoff = int(image_layer_mito.plane.position[0])
        new_binary_data = binary_skeleton.copy()
        new_binary_data[z_cutoff:] = 0
        binary_layer.data = new_binary_data

    image_layer_nucleoid.plane.position = (0, 0, 0)
    image_layer_mito.plane.position = (0, 0, 0)
    animation.capture_keyframe(steps=30)

    image_layer_nucleoid.plane.position = (12, 0, 0)
    image_layer_mito.plane.position = (12, 0, 0)
    animation.capture_keyframe(steps=30)

    image_layer_nucleoid.plane.position = (0, 0, 0)
    image_layer_mito.plane.position = (0, 0, 0)
    animation.capture_keyframe(steps=30)

    image_layer_mito.plane.events.position.connect(replace_binary_data)
    binary_layer.visible = True
    binary_layer.experimental_clipping_planes = [
        {
            "position": (0, 0, 0),
            "normal": (-1, 0, 0),  # point up in z (i.e: show stuff above plane)
        }
    ]

    image_layer_mito.plane.position = (12, 0, 0)
    image_layer_nucleoid.plane.position = (12, 0, 0)
    binary_layer.experimental_clipping_planes[0].position = (12, 0, 0)
    animation.capture_keyframe(steps=60)

    image_layer_mito.visible = False
    image_layer_nucleoid.visible = False
    animation.capture_keyframe()

    splines_layer.visible = True
    animation.capture_keyframe()

    current_step = viewer.dims.current_step
    viewer.dims.current_step = (tracks_um[:, 1].max(), *current_step[1:])
    animation.capture_keyframe(steps=90)

    binary_layer.visible = False
    image_layer_nucleoid.depiction = "volume"
    image_layer_nucleoid.visible = True
    normal_plane_layer.visible = True
    points_layer.visible = True
    animation.capture_keyframe()

    current_step = viewer.dims.current_step
    viewer.dims.current_step = (normal_planes_um[:, :, 0].max(), *current_step[1:])
    animation.capture_keyframe(steps=90)
    viewer.camera.angles = (0, 0, 90)
    animation.capture_keyframe(steps=180)

    image_layer_mito.visible = True
    animation.capture_keyframe(steps=180)

    animation.animate(f"animate_{img_path.stem}.mp4", canvas_only=True)
