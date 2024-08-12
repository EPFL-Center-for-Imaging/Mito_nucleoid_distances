import collections
import logging
import pathlib

import matplotlib.pyplot as plt
import napari
import nellie.im_info.verifier
import nellie.segmentation.filtering
import nellie.segmentation.labelling
import nellie.segmentation.networking
import numpy as np
import pandas as pd
import scipy
import skan
import skimage.io
import splinebox
import tifffile
import tqdm

logger = logging.getLogger()
logger.setLevel(level=logging.ERROR)


def segment_img_with_nellie(
    img_path,
    num_t=None,
    remove_edges=True,
    ch=0,
    output_dirpath=None,
    otsu_thresh_intensity=False,
    dim_sizes=None,
    threshold=None,
    dim_order=None,
):
    file_info = nellie.im_info.verifier.FileInfo(str(img_path.resolve()), output_dir=output_dirpath)
    file_info.find_metadata()
    file_info.load_metadata()
    file_info.change_axes(dim_order)
    file_info.change_dim_res("T", 1)
    for dim in dim_order:
        if dim == "C":
            continue
        file_info.change_dim_res(dim, dim_sizes[dim])
    file_info.change_selected_channel(ch)

    im_info = nellie.im_info.verifier.ImInfo(file_info)

    preprocessing = nellie.segmentation.filtering.Filter(
        im_info, num_t, remove_edges=remove_edges
    )
    preprocessing.run()

    segmenting = nellie.segmentation.labelling.Label(
        im_info, num_t, otsu_thresh_intensity=otsu_thresh_intensity, threshold=threshold
    )
    segmenting.run()

    networking = nellie.segmentation.networking.Network(im_info, num_t)
    networking.run()

    return im_info


def label_img_to_binary_skeleton(label_img, min_length_px):
    """
    min_length_px : int
        Minimum length of sceleton segment in pixels.
        Shorter segments are discarded.
    """
    regionprops = skimage.measure.regionprops_table(label_img, properties=("area",))
    selected_labels = np.where(regionprops["area"] > min_length_px)[0] + 1
    binary_skeleton = np.isin(label_img, selected_labels)
    return binary_skeleton


def skeleton2splines(
    binary_skeleton, pixel_size_um, knots2data_ratio=5, extension_um=0
):
    """
    knots2data : float
        The ration between the number of knots in the spline and
        the data points. The default is 5. This means a skeleton
        segment with 20 pixels would be approximated with a spline
        with 4 knots.
    """
    skeleton = skan.Skeleton(binary_skeleton)

    splines_um = []

    for i in tqdm.tqdm(range(skeleton.n_paths)):
        coordinates = skeleton.path_coordinates(i)
        if len(coordinates) // knots2data_ratio < 4:
            raise RuntimeError(
                "Basis splines of order 3 need at least 4 knots. The minimum number of knots is determined by `min_length_px` // `knots2data_ratio`. Please increase `min_length_px` or decrease `knots2data_ratio` to make sure all splines have at least 4 knots."
            )
        M = len(coordinates) // knots2data_ratio
        coordinates_um = coordinates * pixel_size_um
        spline_um = splinebox.Spline(M=M, basis_function=splinebox.B3(), closed=False)
        spline_um.fit(coordinates_um)
        spline_um = extend_spline(spline_um, extension_um)
        splines_um.append(spline_um)
    return splines_um


def extend_spline(spline, extension):
    knots = np.zeros((spline.M + 2, 3))
    knots[1:-1] = spline.knots
    v_start = spline.eval(0, derivative=1) * -1
    v_start /= np.linalg.norm(v_start)
    knots[0] = knots[1] + v_start * extension

    v_end = spline.eval(spline.M - 1, derivative=1)
    v_end /= np.linalg.norm(v_end)
    knots[-1] = knots[-2] + v_end * extension

    spline = splinebox.Spline(
        M=spline.M + 2, basis_function=spline.basis_function, closed=False
    )
    spline.knots = knots
    return spline


def compute_normal_planes(spline_um, t, pixel_size_um, half_window_size_um=0.5):
    range_um = np.arange(-half_window_size_um, half_window_size_um, pixel_size_um[1])
    ii, jj = np.meshgrid(range_um, range_um)

    deriv_um = spline_um.eval(t, derivative=1)
    normal1_um = np.zeros((len(t), 3))
    normal2_um = np.zeros((len(t), 3))

    normal1_um[:, 1] = deriv_um[:, 2]
    normal1_um[:, 2] = -deriv_um[:, 1]

    normal2_um = np.cross(deriv_um, normal1_um)

    normal1_um /= np.linalg.norm(normal1_um, axis=1)[:, np.newaxis]
    normal2_um /= np.linalg.norm(normal2_um, axis=1)[:, np.newaxis]

    spline_coordinates_um = spline_um.eval(t)

    normal_planes_um = np.multiply.outer(ii, normal1_um) + np.multiply.outer(
        jj, normal2_um
    )
    normal_planes_um = np.rollaxis(normal_planes_um, 2, 0)
    normal_planes_um += spline_coordinates_um[:, np.newaxis, np.newaxis]

    return normal_planes_um


def extract_normal_plane_imgs(normal_planes_um, img, pixel_size_um):
    normal_plane_px = normal_planes_um / pixel_size_um
    shape = normal_plane_px.shape
    vals = scipy.ndimage.map_coordinates(
        img,
        normal_plane_px.reshape(-1, 3).T,
        order=1,
    )
    vals = vals.reshape(shape[:-1]).astype(np.float64)

    # Mask the pixels that are outside the volume
    mask = (
        (np.min(normal_plane_px, axis=3) < 0)
        | (normal_plane_px[:, :, :, 0] > img.shape[0] - 1)
        | (normal_plane_px[:, :, :, 1] > img.shape[1] - 1)
        | (normal_plane_px[:, :, :, 2] > img.shape[2] - 1)
    )
    vals[mask] = np.nan
    return vals


def calculate_distances_to_next(arc_lengths, peak_indices):
    distances = np.zeros(len(t_nucleoids))
    if len(peak_indices) > 1:
        distances[:-1] = np.diff(arc_lengths[peak_indices])
    distances[-1] = np.nan
    return distances


def gaussian(yx, amplitude, yo, xo, sigma_y, sigma_x, theta, offset):
    y, x = yx
    yo = float(yo)
    xo = float(xo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
        4 * sigma_y**2
    )
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()


def calculate_size(imgs):
    sizes = []
    for img in imgs:
        y, x = np.where(~np.isnan(img))
        initial_guess = (
            np.nanmax(img),
            img.shape[0] / 2,
            img.shape[1] / 2,
            2,
            2,
            0,
            0,
        )
        try:
            popt, pcov = scipy.optimize.curve_fit(
                gaussian, (y, x), img[(y, x)], p0=initial_guess
            )
            # Use the relationship between FWHM and the gaussian standard deviation
            sizes.append(2 * np.sqrt(2 * np.log(2)) * np.abs(popt[3:5]))

            # yy, xx = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
            # data_fitted = gaussian((yy, xx), *popt)
            # plt.imshow(img)
            # plt.gca().contour(
            #     xx,
            #     yy,
            #     data_fitted.reshape(*img.shape),
            #     8,
            #     color="w",
            # )
            # plt.show()
        except RuntimeError:
            continue
    return sizes


def plot_distance_histogram(distances_nm, save=None):
    distances_nm = np.array(distances_nm)
    distances_nm = distances_nm[~np.isnan(distances_nm)]
    plt.figure()
    plt.hist(distances_nm, bins=40)
    plt.xlabel("Distance [nm]")
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_size_distribution(sizes_um, save=None):
    sizes_um = np.array(sizes_um)
    sorted_sizes_um = np.sort(sizes_um, axis=1)

    plt.scatter(sorted_sizes_um[:, 1], sorted_sizes_um[:, 1])
    plt.xlabel(r"FWHM [$\mu$m] major axis")
    plt.ylabel(r"FWHM [$\mu$m] minor axis")
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_profile(vals_nucleoids, peak_indices, vals_mitos, path):
    fig, axes = plt.subplots(2, 1, sharex=True)
    for i, vals in enumerate((vals_nucleoids, vals_mitos)):
        axes[i].plot(vals, "x-")
        axes[i].scatter(peak_indices, vals[peak_indices], color="red")
    axes[0].set_ylabel("Nucleoid intensity along spline")
    axes[1].set_ylabel("Mito intensity along spline")
    plt.savefig(path)
    plt.close()


def show_with_napari(
    img, pixel_size_um, splines_um=None, nucleoids_px=None, normal_planes_um=None
):
    viewer = napari.Viewer(ndisplay=3)

    viewer.add_image(
        img,
        channel_axis=1,
        gamma=[2, 1],
        opacity=[1, 0.4],
        contrast_limits=[
            [np.min(img[:, 0]), np.max(img[:, 0]) * 0.5],
            [np.min(img[:, 1]), np.max(img[:, 1])],
        ],
        scale=pixel_size_um,
    )

    if not splines_um is None:
        paths_um = []
        for spline_um in splines_um:
            t = np.linspace(0, spline_um.M - 1, spline_um.M * 10)
            paths_um.append(spline_um.eval(t))

        viewer.add_shapes(
            paths_um,
            shape_type="path",
            edge_width=0.1,
            edge_color="random-path-id",
            edge_colormap="tab10",
        )

    if not nucleoids_px is None:
        nucleoids_um = nucleoids_px * pixel_size_um
        viewer.add_points(np.array(nucleoids_um), opacity=0.1, size=1)

    if not normal_planes_um is None:
        viewer.add_shapes(normal_planes_um, shape_type="rectangle", edge_width=0)

    napari.run()


if __name__ == "__main__":
    pixel_size_um = np.array([0.2, 0.056, 0.056])

    for folder in pathlib.Path("../test_folder").glob("*"):
        # for folder in pathlib.Path(
        #     r"C:/Users/landoni/Desktop/Skeletonization/Mito_nucleoid_distances-main/Mito_nucleoid_distances-main/Test"
        # ).glob("*"):
        img_path = folder / f"{folder.stem}_decon.ome.tiff"

        # Create a folder where the profiles and normal plane
        # images can be saved.
        debug_output_folder = folder / "debug_output"
        debug_output_folder.mkdir(exist_ok=True)

        nellie_output_dir = folder / "nellie_output"
        ch = 1

        segment_img_with_nellie(
            img_path,
            ch=ch,
            remove_edges=False,
            otsu_thresh_intensity=True,
            dim_sizes={
                "T": 1,
                "Z": pixel_size_um[0],
                "Y": pixel_size_um[1],
                "X": pixel_size_um[2],
            },
            dim_order="ZCYX",
            output_dirpath=nellie_output_dir,
        )

        # The path to the label image generated by nellie.
        label_path = nellie_output_dir / f"{img_path.stem}-ch{ch}-im_skel.ome.tif"

        # Load the label image generated by nellie and turn it into a binary skeleton
        label_img = skimage.io.imread(label_path)
        binary_skeleton = label_img_to_binary_skeleton(label_img, min_length_px=20)

        # Turn the segments of the skeleton into splines
        knots2data_ratio = 5
        splines_um = skeleton2splines(
            binary_skeleton,
            pixel_size_um,
            knots2data_ratio=knots2data_ratio,
            extension_um=0.3,
        )

        # Load image
        img = tifffile.imread(img_path)
        # Select the channel with the nucleoids
        nucleoid_img = img[:, 0]
        mito_img = img[:, 1]

        # Create some lists to collect the results
        line_profiles_df_data = collections.defaultdict(lambda: [])
        nucleoids_df_data = collections.defaultdict(lambda: [])

        # In this list we store the planes we want to show
        # in napari.
        selected_normal_planes_um = []

        for branch_id, spline_um in enumerate(splines_um):
            branch_id += 1
            # These are the parameter values of the spline, where
            # the normal planes are calculated. To find the position
            # of the peaks along the spline more accurately,
            # decrease the step size.
            t = np.linspace(0, spline_um.M - 1, spline_um.M * knots2data_ratio * 2)

            # n is the number of data points we will collect for this branch
            n = len(t)

            line_profiles_df_data["Branch_ID"].extend([branch_id] * n)

            # Compute the total length of the spline
            total_arc_length_um = spline_um.arc_length()
            # Convert it to nanometers
            total_arc_length_nm = total_arc_length_um * 1000
            line_profiles_df_data["Branch_total_length_nm"].extend(
                [total_arc_length_nm] * n
            )

            # Compute the length of the spline for each parameter value t
            # (this takes a long time and can probably be sped up exploting the
            # fact that t is monotonically increasing)
            arc_lengths_um = np.array(list(map(spline_um.arc_length, t)))
            arc_lengths_nm = arc_lengths_um * 1000
            line_profiles_df_data["Length_nm"].extend(arc_lengths_nm)

            # In this part we extract the positions of the pixels in the normal planes
            normal_planes_um = compute_normal_planes(
                spline_um, t, pixel_size_um, half_window_size_um=0.5
            )

            # We hold on to the corners of the middle normal plane so we can show
            # them with napari
            middle_idx = normal_planes_um.shape[0] // 2
            selected_normal_planes_um.append(
                normal_planes_um[
                    (
                        np.array((middle_idx, middle_idx, middle_idx, middle_idx)),
                        np.array((0, 0, -1, -1)),
                        np.array((0, -1, -1, 0)),
                    )
                ]
            )

            # Now we have to find the pixel values for each position in the normal planes
            normal_imgs_nucleoid = extract_normal_plane_imgs(
                normal_planes_um, nucleoid_img, pixel_size_um
            )
            normal_imgs_mito = extract_normal_plane_imgs(
                normal_planes_um, mito_img, pixel_size_um
            )

            # Save the normal plane images (comment this line to save time)
            tifffile.imwrite(
                debug_output_folder / f"nucleoid_{branch_id}.tif", normal_imgs_nucleoid
            )
            tifffile.imwrite(
                debug_output_folder / f"mito_{branch_id}.tif", normal_imgs_mito
            )

            # Reduce each normal plane image to a single representative value
            # We use nanmean because pixels of the nomal plane outside the
            # image bounds are nan values.
            intensity_nucleoid = np.nanmean(normal_imgs_nucleoid, axis=(1, 2))
            intensity_mito = np.nanmean(normal_imgs_mito, axis=(1, 2))
            line_profiles_df_data["Intensity_Mito"].extend(intensity_mito)
            line_profiles_df_data["Intensity_Nucleoid"].extend(intensity_nucleoid)

            # Find the peaks in the along the plane
            # (check the find_peaks documentation to filter the peaks in a biological
            # meaningful way).
            peak_indices, _ = scipy.signal.find_peaks(
                intensity_nucleoid, prominence=200
            )

            # Creat an array with N and Y for the csv file
            peak_yn = np.array(["N"] * n)
            peak_yn[peak_indices] = "Y"
            line_profiles_df_data["Peak_YN"].extend(peak_yn)

            # Create the nucleoid_id column for the csv file for this branch
            nucleoid_id = np.array([np.nan] * n)
            nucleoid_id[peak_indices] = np.cumsum(np.ones(len(peak_indices)))
            line_profiles_df_data["Nucleoid_ID"].extend(nucleoid_id)

            # Save the profile plot (comment this line to save time)
            plot_profile(
                intensity_nucleoid,
                peak_indices,
                intensity_mito,
                debug_output_folder / f"{branch_id}.png",
            )

            # The number of nucleoids detected on this branch
            n_nucleoids = len(peak_indices)

            # Process the peak(s) if the current branch has at least one
            if n_nucleoids > 0:
                # Parameter values of the splines where the peak(s) are located
                t_nucleoids = t[peak_indices]

                # Add the branch id, nucleoid_id and length to the data for the nucleoid csv
                nucleoids_df_data["Branch_ID"].extend([branch_id] * n_nucleoids)
                nucleoids_df_data["Nucleoid_ID"].extend(np.arange(n_nucleoids) + 1)
                nucleoids_df_data["Length_nm"].extend(arc_lengths_nm[peak_indices])

                # To calculate the distance to the next nucleoid we can use the fact
                # that we have already computed the length for all paramters values t.
                distances_to_next_nm = calculate_distances_to_next(
                    arc_lengths_nm, peak_indices
                )
                nucleoids_df_data["Distance_to_next_nm"].extend(distances_to_next_nm)

                # Location of the peaks in micrometers
                nucleoids_um = spline_um.eval(t_nucleoids)
                if nucleoids_um.ndim == 1:
                    # If there is only one peak, we have to add an axis to make
                    # things compatible.
                    nucleoids_um = nucleoids_um[np.newaxis, :]
                # Save the pixel coordinates of each nucleoid to the data for the nucleoid csv
                # so they can easily be found using Fiji.
                nucleoids_df_data["z_px"].extend(nucleoids_um[:, 0] / pixel_size_um[0])
                nucleoids_df_data["y_px"].extend(nucleoids_um[:, 1] / pixel_size_um[1])
                nucleoids_df_data["x_px"].extend(nucleoids_um[:, 2] / pixel_size_um[2])

        # Save the csv files
        df = pd.DataFrame(nucleoids_df_data)
        df["Filename"] = img_path.name
        df.to_csv(folder / f"{img_path.stem}_Nucleoids.csv")
        df = pd.DataFrame(line_profiles_df_data)
        df["Filename"] = img_path.name
        df.to_csv(folder / f"{img_path.stem}_LineProfiles.csv")

        # Plot and save the distance histogram
        plot_distance_histogram(
            nucleoids_df_data["Distance_to_next_nm"],
            save=folder / f"{img_path.stem}_distances.pdf",
        )

        # Showing the results with napari is useful for debugging but slow
        # so you can comment this part once you are happy with the results.
        nucleoids_px = np.stack(
            [
                nucleoids_df_data["z_px"],
                nucleoids_df_data["y_px"],
                nucleoids_df_data["x_px"],
            ],
            axis=-1,
        )
        show_with_napari(
            img,
            pixel_size_um,
            splines_um,
            nucleoids_px,
            selected_normal_planes_um,
        )
