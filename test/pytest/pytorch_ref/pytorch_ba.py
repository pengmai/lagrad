import torch

# BA global parameters
BA_NCAMPARAMS = 11  # number of camera parameters for BA
ROT_IDX = 0
C_IDX = 3
F_IDX = 6
X0_IDX = 7
RAD_IDX = 9

######### BA objective in Python (torch) #############


def rodrigues_rotate_point(rot, X):
    sqtheta = torch.sum(rot ** 2)
    if sqtheta != 0.0:
        theta = torch.sqrt(sqtheta)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        theta_inverse = 1.0 / theta

        w = theta_inverse * rot
        w_cross_X = torch.cross(w, X)
        tmp = torch.dot(w, X) * (1.0 - costheta)

        return X * costheta + w_cross_X * sintheta + w * tmp
    else:
        return X + torch.cross(rot, X)


def radial_distort(rad_params, proj):
    rsq = torch.sum(proj ** 2)
    L = 1.0 + rad_params[0] * rsq + rad_params[1] * rsq * rsq
    return proj * L


def project(cam, X):
    Xcam = rodrigues_rotate_point(
        cam[ROT_IDX : ROT_IDX + 3], X - cam[C_IDX : C_IDX + 3]
    )
    distorted = radial_distort(cam[RAD_IDX : RAD_IDX + 2], Xcam[0:2] / Xcam[2])
    return distorted * cam[F_IDX] + cam[X0_IDX : X0_IDX + 2]


def compute_reproj_err(cam, X, w, feat):
    return w * (project(cam, X) - feat)


def compute_w_err(w):
    return 1.0 - w ** 2
