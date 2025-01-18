import torch
import torch.nn.functional as F

def create_temporal_attention_mask(batch_size, frames=9, height=30, width=45, kernel_size=3, decay_factor=0.5):
    """
    Args:
    batch_size: Number of batches
    frames: Number of frames
    height: Height of each frame
    width: Width of each frame
    kernel_size: Size of the window for neighboring frames to consider
    decay_factor: Decay factor for neighboring frames, lower values reduce weight faster

    Returns:
    attention_mask: (batch_size, sequence_length, sequence_length)
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd to have a central frame."

    num_tokens_per_frame = height * width
    sequence_length = frames * num_tokens_per_frame

    # Step 1: Create frame-level attention weights (frames, frames)
    half_window = kernel_size // 2
    weights = torch.tensor([
        decay_factor ** abs(i - half_window) for i in range(kernel_size)
    ])
    
    frame_attention = torch.zeros(frames, frames)
    for i in range(frames):
        for j in range(max(0, i - half_window), min(frames, i + half_window + 1)):
            frame_attention[i, j] = weights[half_window + j - i]

    # Step 2: Normalize frame-level weights
    # frame_attention /= frame_attention.sum(dim=-1, keepdim=True)

    # Step 3: Expand to token level (sequence_length, sequence_length)
    token_attention = frame_attention.repeat_interleave(num_tokens_per_frame, dim=0)
    token_attention = token_attention.repeat_interleave(num_tokens_per_frame, dim=1)

    # Step 4: Expand to batch level (batch_size, sequence_length, sequence_length)
    attention_mask = token_attention.unsqueeze(0).repeat(batch_size, 1, 1)

    return attention_mask

def create_attention_mask(batch_size, frames=9, height=30, width=45):
    """
    Args:
    batch, frames, height, width
    
    returns:
    attention_mask: (batch, sequence_length, sequen_length)
    """
    num_tokens_per_frame = height * width
    sequence_length = frames * num_tokens_per_frame

    # Step 1: frame level (frames, frames)
    frame_causal_mask = torch.triu(torch.ones(frames, frames), diagonal=1)
    frame_causal_mask = frame_causal_mask.masked_fill(frame_causal_mask == 1, float('-inf')).masked_fill(frame_causal_mask == 0, 0)

    import pdb; pdb.set_trace()


    # Step 2: token level (sequence_length, sequence_length)
    frame_causal_mask = frame_causal_mask.repeat_interleave(num_tokens_per_frame, dim=0)
    frame_causal_mask = frame_causal_mask.repeat_interleave(num_tokens_per_frame, dim=1)

    # Step 3: batch level (batch_size, sequence_length, sequence_length)
    attention_mask = frame_causal_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    
    return attention_mask


# Example usage
batch_size = 2
frames = 13
height = 30
width = 45
kernel_size = frames
decay_factor = 0.8

attention_mask = create_temporal_attention_mask(batch_size, frames, height, width, kernel_size, decay_factor)

# attention_mask = create_temporal_attention_mask(batch_size, frames, height, width)

print(attention_mask.shape)
