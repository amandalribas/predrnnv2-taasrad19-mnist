def validate_loaded_sequence_length(
    seq_len: int, input_seq_len: int, output_seq_len: int
):
    if input_seq_len + output_seq_len > seq_len:
        raise Exception(
            f"There aren't any images left ({seq_len - input_seq_len}) to fulfill the output sequence length (={output_seq_len})"
        )


def prepare_for_predrnn(data):
    return data.permute(0, 1, 3, 4, 2).contiguous()


def reshape_patch(frames, patch_size):
    batch, seq, height, width, channel = frames.shape

    patch_size = int(patch_size)

    assert height % patch_size == 0
    assert width % patch_size == 0

    new_height = height // patch_size
    new_width = width // patch_size

    frames = frames.reshape(
        batch,
        seq,
        new_height,
        patch_size,
        new_width,
        patch_size,
        channel,
    )

    frames = frames.permute(0, 1, 2, 4, 3, 5, 6).contiguous()

    frames = frames.reshape(
        batch,
        seq,
        new_height,
        new_width,
        patch_size * patch_size * channel,
    )

    return frames


def reshape_patch_back(patches, patch_size, img_channel):
    batch, seq, patch_height, patch_width, patch_channel = patches.shape

    patch_size = int(patch_size)

    frames = patches.reshape(
        batch,
        seq,
        patch_height,
        patch_width,
        patch_size,
        patch_size,
        img_channel,
    )

    frames = frames.permute(
        0, 1, 2, 4, 3, 5, 6
    ).contiguous()

    frames = frames.reshape(
        batch,
        seq,
        patch_height * patch_size,
        patch_width * patch_size,
        img_channel,
    )

    return frames