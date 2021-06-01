import os
from any_dnn_model import validate_model
from utility import parse_multiple_sources_images
from annotation import annotation


def main():
    model_file_name = r"C:\Users\joexi\work\element\3d_rendering\Assets\25_6a765b18-60bd-439c-a03f-295edd9d4b09_v0_model_rr_net_converted.pb"

    sorted_steps, dict_data = parse_multiple_sources_images()

    for step in sorted_steps:
        file_a, frame_a, video_id_a, file_id_a = dict_data[step][0]
        file_b, frame_b, video_id_b, file_id_b = dict_data[step][1]

        # validate_model(file_b, model_file_name)
        if step == 15:
            annotation(file_a)
            annotation(file_b)
            break
        else:
            continue


if __name__ == "__main__":
    main()
