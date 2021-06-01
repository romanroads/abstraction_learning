import glob
import os
from aes_cipher import AESCipher


def parse_multiple_sources_images():
    img_files = glob.glob(os.path.join("../sample_data", "*.png"))
    video_header_token = "video"
    frame_header_token = "frame"
    step_header_token = "step"
    file_header_token = "fid"

    steps = []
    dict_step_file_name = {}

    for img_file in img_files:
        basename = os.path.basename(img_file)
        video_token, video_id, step_token, step, file_token, file_id, frame_token, frame_id =\
            basename.split(".")[0].split("_")

        if video_token == video_header_token and frame_token == frame_header_token and step_token == step_header_token\
                and file_token == file_header_token:
            video_id = int(video_id)
            frame_id = int(frame_id)
            file_id = int(file_id)
            step = int(step)
            steps.append(step)

            if step not in dict_step_file_name:
                dict_step_file_name[step] = {}

            dict_step_file_name[step][video_id] = (img_file, frame_id, video_id, file_id)

    sorted_step = sorted(set(steps))

    return sorted_step, dict_step_file_name


def load_graph(frozen_graph_filename, output_file_for_tensorboard=False, print_constant_tensor_value=False,
               is_decryption_needed=False):

    import tensorflow.compat.v1 as tfc

    KEY = "XFKN433NIWRR67SQNGZR1U3DP251M3"
    with tfc.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tfc.GraphDef()
        file_byte_data = f.read()

        if is_decryption_needed:
            print('key:', KEY)
            cipher = AESCipher(KEY)
            file_byte_data = cipher.decrypt(file_byte_data)

        graph_def.ParseFromString(file_byte_data)

    with tfc.Graph().as_default() as graph:
        # Note: The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tfc.import_graph_def(graph_def, name="")

    if print_constant_tensor_value:
        # Note: printing some constant tensor values
        graph_nodes = [n for n in graph_def.node]
        wts = [n for n in graph_nodes if n.op == 'Const']
        for n in wts:
            if "image" in n.name:
                print("Name of the node - %s" % n.name)
                print("Value - ")
                print(tensor_util.MakeNdarray(n.attr['value'].tensor))

    if output_file_for_tensorboard:
        # Note: save summary files for tensorboard
        tfc.disable_eager_execution()
        writer = tfc.summary.FileWriter("logs")
        writer.add_graph(graph)
        writer.flush()
        writer.close()

    return graph


def find_label_file_name(image_file_name):
    file_name = os.path.basename(image_file_name).replace(".png", ".txt")
    path_text = os.path.dirname(image_file_name)

    return os.path.join(path_text, file_name)
