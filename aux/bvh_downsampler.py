import numpy as np
import re

def parse_bvh_file(file_path):
    frames = []
    with open(file_path, 'r') as bvh_file:
        lines = bvh_file.readlines()

        # Find the Frames section and parse the number of frames
        frames_line = [line for line in lines if line.startswith('Frames:')]
        if frames_line:
            num_frames = int(frames_line[0].split()[1])
            print('Number of frames:', num_frames)
        else:
            raise ValueError("Frames section not found in the BVH file.")

        # Calculate the new frame count
        new_num_frames = num_frames // 4

        # Find the starting index of the Frames data
        frames_number_index = next((i for i, line in enumerate(lines) if re.match(r'^Frames\b', line)), None)
        if frames_number_index is None:
            raise ValueError("Frames data not found in the BVH file.")
        frames_data_start_index = frames_number_index + 2

        for i in range(new_num_frames):
            # assumes 120 fps (downsamples to 30 fps)
            frame_index = frames_data_start_index + (i * 4)
            frame_data = lines[frame_index]
            frames.append(frame_data)

    return frames, frames_number_index


def write_bvh_file(original_file_path, file_path, frames, frames_number_index):
    with open(file_path, 'w') as bvh_file:
        with open(original_file_path, 'r') as original_file:
            lines = original_file.readlines()

            # Calculate the new frame count
            new_num_frames = len(frames)

            line = lines[frames_number_index].strip()
            tokens = line.split()
            print(f"tokens: {tokens}")

            # Update the desired token
            tokens[1] = str(new_num_frames) + '\n'

            # Join the tokens back into a line
            updated_line = ' '.join(tokens)

            # Update the line in the list
            lines[frames_number_index] = updated_line

            # Find the Frame Time section
            line = lines[frames_number_index + 1].strip()
            tokens = line.split()
            print(f"tokens: {tokens}")
            tokens[2] = str(1 / 30) + '\n'
            updated_line = ' '.join(tokens)
            lines[frames_number_index + 1] = updated_line

            # Get the starting index of the Frames data
            frames_number_index = frames_number_index + 2

            lines_pre_frames = [line for line in lines[:frames_number_index]]
            lines_new_frames = frames
            lines_pre_frames.extend(lines_new_frames)
            bvh_file.writelines(lines_pre_frames)

if __name__ == '__main__':
    original_file_path = '/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity/data' \
                         '/cmu_motions' \
                         '/walking_136_602.bvh'
    parsed_frames, frames_number_index = parse_bvh_file(original_file_path)
    new_file_path = '/Users/bendiksen/Desktop/tmp.bvh'
    write_bvh_file(original_file_path, new_file_path, parsed_frames, frames_number_index)