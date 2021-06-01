import os
import cv2
import numpy as np
from utility import parse_multiple_sources_images, find_label_file_name


def get_poly(line):
    points = line.split()
    agent_id = int(points[0])
    num_points = int(points[1])
    num_key_points = 0 if len(points) <= num_points * 2 + 3 else int(points[num_points * 2 + 3])
    if num_key_points <= 0:
        assert len(points) == num_points * 2 + 3,\
            "[ERROR] utils: data entry %s seems to be having corrupted structures" % line
    else:
        assert len(points) == num_points * 2 + 3 + num_key_points * 3 + 1, \
            "[ERROR] utils: data entry %s seems to be having corrupted structures" % line

    poly = []
    feature_points = []
    max_x, max_y = 0, 0
    min_x, min_y = np.inf, np.inf

    # Note a polygon must be defined with at least 3 points
    if num_points < 3:
        print("[WARNING] mistakes in the annotation, line %s num points %s for polygon" % (line, num_points))
        return None

    for i in range(num_points):
        x = float(points[2 * i + 2])
        y = float(points[2 * i + 3])
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        poly.append((x, y))

    for i in range(num_key_points):
        x = float(points[3 * i + num_points * 2 + 3 + 1])
        y = float(points[3 * i + num_points * 2 + 3 + 2])
        visible = float(points[3 * i + num_points * 2 + 3 + 3])
        feature_points.append((x, y, visible))

    # Note: when the text annotation file contains labels for feature points, but the constants defines
    # empty or zero feature points, that means we do not wanna train using feature point data, reset it to null
    num_key_points_for_single_class_detection = 0
    if num_key_points_for_single_class_detection <= 0:
        feature_points = []

    category_name = points[num_points * 2 + 3 - 1]
    return agent_id, poly, [min_x, min_y, max_x, max_y], category_name, feature_points


def plot_frame_id(image, frame_number, video_id, file_id, font_scale=5.0, line_thickness=15):
    offset_width = 200
    offset_height = 200
    RED = (0, 0, 255)
    cv2.putText(image, "Data Source %s - Frame %s" % (video_id, frame_number),
                (offset_width, offset_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                RED,
                line_thickness)


def check_annotation(list_text_files_combo):
    WHITE = (255, 255, 255)
    BLUE_BRG = (255, 0, 0)
    BLACK = (0, 0, 0)
    window_name = "demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 0, 0)
    cv2.resizeWindow(window_name, 1920, 1080)

    width_per_frame = 3840
    height_per_frame = 2160
    num_frame = len(list_text_files_combo)
    agg_width = width_per_frame * num_frame
    agg_height = height_per_frame * 2
    frame_aggregated = np.full((agg_height, agg_width, 3), BLACK, dtype=np.uint8)
    dict_class_names_color = {"Car": (102, 0, 0),
                              "Building": (244, 10, 120),
                              "Pedestrian": (120, 255, 20),
                              "Motorcycle": (123, 45, 1),
                              "RoadFeatures": (102, 51, 102),
                              "Tree": (0, 255, 0),
                              "AV": (10, 122, 153)}

    for i in range(num_frame):
        text_file, frame_id, video_id, file_id = list_text_files_combo[i]
        image_file = text_file.replace(".txt", ".png")
        frame = cv2.imread(image_file)
        ori_height, ori_width, _ = frame.shape

        frame_for_all_objects = np.full((ori_height, ori_width, 3), BLACK, dtype=np.uint8)

        line_width = 4
        text_font = 1
        dot_size = 12

        with open(text_file) as f:
            line = f.readline().strip()
            while line:
                parsed_annotation = get_poly(line)
                if parsed_annotation is None:
                    print("[WARNING] found a mistake in the annotation file %s, line %s" % (text_file, line))
                    line = f.readline().strip()
                    continue

                agent_id, poly, box, category_name, key_points = parsed_annotation

                if category_name in dict_class_names_color:
                    color_code = dict_class_names_color[category_name]
                else:
                    color_code = BLUE_BRG

                black_mask_for_objects = np.zeros((ori_height, ori_width), dtype=np.uint8)
                frame_for_objects = np.full((ori_height, ori_width, 3), color_code, dtype=np.uint8)

                arr_pixel_cords = np.array(poly, dtype=np.int32)

                x_ave, y_ave = int(np.mean(arr_pixel_cords[:, 0])), int(np.mean(arr_pixel_cords[:, 1]))

                poly_mask_cords = np.array(poly, dtype=np.int32)

                black_bkg_mask_fg_i = np.zeros((ori_height, ori_width), dtype=np.uint8)
                cv2.fillPoly(black_bkg_mask_fg_i, [poly_mask_cords], WHITE)
                black_mask_for_objects = black_bkg_mask_fg_i | black_mask_for_objects

                frame_for_objects = cv2.bitwise_and(frame_for_objects, frame_for_objects, mask=black_mask_for_objects)

                alpha_for_this_object = 1.0

                if category_name == "RoadFeatures":
                    alpha_for_this_object = 0.4

                frame_for_all_objects = cv2.addWeighted(frame_for_objects, alpha_for_this_object, frame_for_all_objects, 1, 0)

                line = f.readline().strip()

        alpha = 0
        frame = cv2.addWeighted(frame_for_all_objects, alpha, frame, 1, 0)
        w_s = 0 + i * ori_width
        w_e = w_s + ori_width
        r_s = 0
        r_e = ori_height

        plot_frame_id(frame, frame_id, video_id, file_id)
        plot_frame_id(frame_for_all_objects, frame_id, video_id, file_id)
        frame_aggregated[r_s:r_e, w_s:w_e] = frame
        frame_aggregated[r_e:r_e + ori_height, w_s:w_e] = frame_for_all_objects

    cv2.imshow(window_name, frame_aggregated)
    cv2.imwrite("demo.png", frame_aggregated)

    while True:
        key = cv2.waitKey(1) & 0xFF
        # Esc key pressed or window closed?
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(window_name)
            break


def main():

    sorted_steps, dict_data = parse_multiple_sources_images()

    for step in sorted_steps:
        file_a, frame_a, video_id_a, file_id_a = dict_data[step][0]
        file_b, frame_b, video_id_b, file_id_b = dict_data[step][1]

        label_file_a = find_label_file_name(file_a)
        label_file_b = find_label_file_name(file_b)

        if os.path.exists(label_file_a) and os.path.exists(file_a):
            check_annotation([(label_file_a, frame_a, video_id_a, file_id_a),
                              (label_file_b, frame_b, video_id_b, file_id_b)])


if __name__ == "__main__":
    main()
