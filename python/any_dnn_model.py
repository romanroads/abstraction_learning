import cv2
import tensorflow as tf
import tensorflow.compat.v1 as tfc
from tensorflow.compat.v1 import ConfigProto
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from utility import load_graph


def validate_model(path, model_path):
    WINDOW_NAME = "perception_frontend"
    WIDTH_WINDOW = 800
    HEIGHT_WINDOW = 600
    IMAGE_WIDTH_ENDTER_DNN = 800
    IMAGE_HEIGHT_ENDTER_DNN = 600
    WINDOW_START_WIDTH = 10
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(WINDOW_NAME, WIDTH_WINDOW, HEIGHT_WINDOW)
    cv2.moveWindow(WINDOW_NAME, WINDOW_START_WIDTH, 10)
    BLUE_BRG = [255, 0, 0]
    WHITE = (255, 255, 255)
    NUM_FEATURE_POINTS = 0  # number of feature points, can be 3 points for head, middle and tail
    NUM_MASK_POINTS = 28  # 28 x 28
    NUM_FEATURE_POINT_MASK_DIMEN = 56
    FIXED_LEN_OF_AGENTS = 100
    THRESHOLD_CONFIDENCE_SCORE = 0.8

    with tf.device("/GPU:0"):
        graph = load_graph(model_path, is_decryption_needed=True)
        hybrid_box_fixed_len = graph.get_tensor_by_name('hybrid_boxes_fixed_len:0')

        frame = cv2.imread(path)

        frame = cv2.resize(frame, (IMAGE_WIDTH_ENDTER_DNN, IMAGE_HEIGHT_ENDTER_DNN))
        desired_height, desired_width, _ = frame.shape
        black_mask_for_objects = np.zeros((desired_height, desired_width), dtype=np.uint8)
        frame_for_objects = np.full((desired_height, desired_width, 3), BLUE_BRG, dtype=np.uint8)

        config = ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        config.gpu_options.visible_device_list = '0'
        config.allow_soft_placement = False

        with tfc.Session(graph=graph, config=config) as sess:
            input_data = frame.flatten()
            hybrid_box_fixed_len_arr \
                = sess.run([hybrid_box_fixed_len],
                                feed_dict={
                                    "input:0": input_data  # x : frame should be used if using the original input,
                                    # but we assume we always convert original input layer to our non-shaped input for ML
                                    # net that has bugs for shape input tensor, has to be [None,]
                                })
            hybrid_box_fixed_len_arr = hybrid_box_fixed_len_arr[0]
            num_float_in_output_tensor = len(hybrid_box_fixed_len_arr)
            num_float_per_agent = int(num_float_in_output_tensor / FIXED_LEN_OF_AGENTS)
            hybrid_box_fixed_len_arr = np.reshape(hybrid_box_fixed_len_arr, (FIXED_LEN_OF_AGENTS, num_float_per_agent))

            for i in range(FIXED_LEN_OF_AGENTS):
                list_of_mask_points_i = []
                x0, y0, x1, y1, score_i, label_i = hybrid_box_fixed_len_arr[i][0:6]
                x_range = int(round(x1 - x0))
                y_range = int(round(y1 - y0))
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

                mask_i_empty = np.zeros((desired_height, desired_width), bool)
                mask_i = hybrid_box_fixed_len_arr[i][6:6+NUM_MASK_POINTS*NUM_MASK_POINTS].reshape((NUM_MASK_POINTS,
                                                                                                   NUM_MASK_POINTS))

                feature_points_i_new = hybrid_box_fixed_len_arr[i][6+NUM_MASK_POINTS*NUM_MASK_POINTS:]
                if len(feature_points_i_new) == NUM_FEATURE_POINTS:
                    feature_points_i_new = feature_points_i_new.reshape(NUM_FEATURE_POINTS, 1)
                    feature_points_i = feature_points_i_new

                # Note these are padded value to make the output fixed length
                if score_i <= -1:
                    break

                if score_i < THRESHOLD_CONFIDENCE_SCORE:
                    continue

                binary_mask_i = mask_i > 0.5

                for feature_point_index in range(NUM_FEATURE_POINTS):
                    if feature_point_index == 0:
                        color_code = (0, 0, 255)
                    elif feature_point_index == 1:
                        color_code = (0, 255, 0)
                    else:
                        color_code = (0, 0, 0)

                    dimen_feature_point_mask_array = NUM_FEATURE_POINT_MASK_DIMEN
                    max_index = feature_points_i[feature_point_index]
                    max_row_index = int(max_index / dimen_feature_point_mask_array)
                    max_col_index = max_index % dimen_feature_point_mask_array

                    frac_row = max_row_index * 1. / dimen_feature_point_mask_array
                    frac_col = max_col_index * 1. / dimen_feature_point_mask_array
                    x_feature_point = int(x0 + frac_col * x_range)
                    y_feature_point = int(y0 + frac_row * y_range)
                    cv2.circle(frame, (x_feature_point, y_feature_point), 5,
                               color_code,
                               -1)

                seg_map = SegmentationMapsOnImage(binary_mask_i, shape=binary_mask_i.shape)
                seg_map = seg_map.resize((y_range, x_range))
                scaled_mask = seg_map.get_arr()

                mask_i_empty[y0:y0+y_range, x0:x0+x_range] = scaled_mask

                mask_indices = np.argwhere(mask_i_empty == True).astype(float)
                mask_indices = mask_indices.astype(int)

                for point in mask_indices:
                    list_of_mask_points_i.append((point[1], point[0]))

                poly_mask_cords = np.array(list_of_mask_points_i, dtype=np.int32)

                black_bkg_mask_fg_i = np.zeros((desired_height, desired_width), dtype=np.uint8)
                cv2.fillPoly(black_bkg_mask_fg_i, [poly_mask_cords], WHITE)

                black_mask_for_objects = black_bkg_mask_fg_i | black_mask_for_objects

                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

        frame_for_objects = cv2.bitwise_and(frame_for_objects, frame_for_objects, mask=black_mask_for_objects)

        alpha = 0.7
        frame = cv2.addWeighted(frame_for_objects, alpha, frame, 1, 0)

        cv2.imshow(WINDOW_NAME, frame)

        while True:
            key = cv2.waitKey(1) & 0xFF
            # Esc key pressed or window closed?
            if key == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyWindow(WINDOW_NAME)
                break

