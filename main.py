import logging
import pathlib

import nellie.im_info.im_info
import nellie.segmentation.filtering
import nellie.segmentation.labelling
import nellie.segmentation.networking

logger = logging.getLogger()
logger.setLevel(level=logging.ERROR)

import matplotlib.pyplot as plt
import napari
import numpy as np
import scipy
import skan
import skimage.io
import splinebox
import tifffile
import tqdm


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
    im_info = nellie.im_info.im_info.ImInfo(
        str(img_path.resolve()),
        ch=ch,
        output_dirpath=str(output_dirpath),
        dim_sizes=dim_sizes,
        dimension_order=dim_order,
    )

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


def skeleton2splines(label_path, min_length_px, knots2data_ratio=5, extension_um=0):
    """
    min_length_px : int
        Minimum length of sceleton segment in pixels.
        Shorter segments are discarded.
    knots2data : float
        The ration between the number of knots in the spline and
        the data points. The default is 5. This means a skeleton
        segment with 20 pixels would be approximated with a spline
        with 4 knots.
    """
    if min_length_px // knots2data_ratio < 4:
        raise RuntimeError(
            f"Basis splines of order 3 need at least 4 knots. You have specified `min_length_px`={min_length_px} and `knots2data_ratio`={knots2data_ratio}. This results in a minimum number of knots of `min_length_px` // `knots2data_ratio` = {min_length_px // knots2data_ratio}. Please increase `min_length_px` or decrease `knots2data_ratio` to make sure all splines have at least 4 knots."
        )

    label_img = skimage.io.imread(label_path)

    regionprops = skimage.measure.regionprops_table(label_img, properties=("area",))
    selected_labels = np.where(regionprops["area"] > min_length_px)[0] + 1

    binary_skeleton = np.isin(label_img, selected_labels)

    skeleton = skan.Skeleton(binary_skeleton)

    splines_um = []

    for i in tqdm.tqdm(range(skeleton.n_paths)):
        coordinates = skeleton.path_coordinates(i)
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


def extract_normal_plane_imgs(
    spline_um, t, img, pixel_size_um, half_window_size_um=0.5
):
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

    normal_plane_um = np.multiply.outer(ii, normal1_um) + np.multiply.outer(
        jj, normal2_um
    )
    normal_plane_um = np.rollaxis(normal_plane_um, 2, 0)
    normal_plane_um += spline_coordinates_um[:, np.newaxis, np.newaxis]

    normal_plane_px = normal_plane_um / pixel_size_um
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


def calculate_distances(spline_um, t_peaks):
    distances_um = []
    if len(t_peaks) > 1:
        for t0, t1 in zip(t_peaks[:-1], t_peaks[1:]):
            distance_um = spline_um.arc_length(t0, t1)
            distances_um.append(distance_um)
    return distances_um


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


def plot_distance_histogram(distances_um, save=None):
    plt.figure()
    plt.hist(distances_um, bins=40)
    plt.xlabel(r"Distance [$\mu$m]")
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


def plot_profile(vals, peak_indices, path):
    plt.plot(vals, "x-")
    plt.scatter(peak_indices, vals[peak_indices], color="red")
    plt.savefig(path)
    plt.close()


def show_with_napari(img, pixel_size_um, splines_um, peaks_um):
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

    viewer.add_points(np.array(peaks_um), opacity=0.1, size=1)

    napari.run()


if __name__ == "__main__":
    pixel_size_um = np.array([0.2, 0.056, 0.056])

    for folder in pathlib.Path("../test_folder").glob("*"):
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

        # Turn the segments of the skeleton into splines
        knots2data_ratio = 5
        splines_um = skeleton2splines(
            label_path,
            min_length_px=20,
            knots2data_ratio=knots2data_ratio,
            extension_um=0.3,
        )

        # Load image
        img = tifffile.imread(img_path)
        # Select the channel with the nucleoids
        nucleoid_img = img[:, 0]

        # Create some lists to collect the results
        all_distances_um = []
        all_peaks_um = []
        all_sizes_um = []

        for i, spline_um in enumerate(splines_um):
            # These are the parameter values of the spline, where
            # the normal planes are calculated. To find the position
            # of the peaks along the spline more accurately,
            # decrease the step size.
            t = np.linspace(0, spline_um.M - 1, spline_um.M * knots2data_ratio * 2)

            normal_imgs = extract_normal_plane_imgs(
                spline_um, t, nucleoid_img, pixel_size_um, half_window_size_um=0.5
            )

            # Save the normal plane images (comment this line to save time)
            tifffile.imwrite(debug_output_folder / f"{i}.tif", normal_imgs)

            # Reduce each normal plane image to a single representative value
            # We use nanmean because pixels of the nomal plane outside the
            # image bounds are nan values.
            vals = np.nanmean(normal_imgs, axis=(1, 2))

            # Find the peaks in the along the plane
            # (check the find_peaks documentation to filter the peaks in a biological
            # meaningful way).
            peak_indices, _ = scipy.signal.find_peaks(vals, prominence=200)

            # Save the profile plot (comment this line to save time)
            plot_profile(vals, peak_indices, debug_output_folder / f"{i}.tif")

            # Calculate the size of the nucleoid at each peak
            sizes_px = calculate_size(normal_imgs[peak_indices])
            if len(sizes_px) > 0:
                sizes_um = sizes_px * pixel_size_um[1:]
                all_sizes_um.extend(sizes_um)

            # Process the peak(s) if the current section has at least one
            if len(peak_indices) > 0:
                # Parameter values of the splines where the peak(s) are located
                t_peaks = t[peak_indices]

                # Location of the peaks in micrometers
                peaks_um = spline_um.eval(t_peaks)

                if peaks_um.ndim == 1:
                    # If there is only one peak, we have to add an axis to make
                    # things compatible.
                    peaks_um = peaks_um[np.newaxis, :]

                # Collecte the peak locations so they can be save as a csv later.
                all_peaks_um.extend(peaks_um)

                distances_um = calculate_distances(spline_um, t_peaks)

                # Collecte the distances so they can be save as a csv later.
                all_distances_um.extend(distances_um)

        plot_distance_histogram(all_distances_um, save=folder / "distances.pdf")

        plot_size_distribution(all_sizes_um, save=folder / "sizes.pdf")

        # Showing the results with napari is useful for debugging but slow
        # so you can comment this once you are happy with the results.
        show_with_napari(img, pixel_size_um, splines_um, all_peaks_um)
