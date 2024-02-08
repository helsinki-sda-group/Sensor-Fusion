import torch

def eulerAnglesToRotationMatrix(theta):
    batch_size = theta.shape[0]
    one = torch.ones(batch_size, dtype=theta.dtype, device=theta.device)
    zero = torch.zeros(batch_size, dtype=theta.dtype, device=theta.device)

    R_x = torch.stack(
        [one, zero, zero, zero, torch.cos(theta[:, 0]), -torch.sin(theta[:, 0]), zero, torch.sin(theta[:, 0]), torch.cos(theta[:, 0])],
        dim=-1,
    ).view(batch_size, 3, 3)

    R_y = torch.stack(
        [torch.cos(theta[:, 1]), zero, torch.sin(theta[:, 1]), zero, one, zero, -torch.sin(theta[:, 1]), zero, torch.cos(theta[:, 1])],
        dim=-1,
    ).view(batch_size, 3, 3)

    R_z = torch.stack(
        [torch.cos(theta[:, 2]), -torch.sin(theta[:, 2]), zero, torch.sin(theta[:, 2]), torch.cos(theta[:, 2]), zero, zero, zero, one],
        dim=-1,
    ).view(batch_size, 3, 3)

    R = torch.matmul(R_z, torch.matmul(R_y, R_x))
    return R

def pose_6DoF_to_matrix(pose):
    R = eulerAnglesToRotationMatrix(pose[:, :3])
    t = pose[:, 3:].view(-1, 3, 1)
    bottom_row = torch.tensor([0, 0, 0, 1], dtype=pose.dtype, device=pose.device).view(1, 1, 4).repeat(pose.shape[0], 1, 1)
    R_t = torch.cat((R, t), dim=-1)
    R_t = torch.cat((R_t, bottom_row), dim=1)
    return R_t

def pose_accu(Rt_pre, R_rel):
    Rt_rel = pose_6DoF_to_matrix(R_rel)
    return torch.bmm(Rt_pre, Rt_rel)

def seq_accu(pose):
    """
    Generate the global pose matrices from a series of relative poses
    """
    answer = torch.stack([torch.eye(4, device=pose.device) for _ in range(pose.size(0))])
    for i in range(pose.size(1)):
        answer = pose_accu(answer, pose[:, i])
    return answer