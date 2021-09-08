import torch
import torch.nn.functional as F
import random
import numpy as np



def EPE_Loss(input_flow, target_flow, mean=True):

    batch_size = target_flow.size(0)
    # target_flow = torch.rand(batch_size, 2, 460, 620)
    # #target_flow = target_flow.cuda()    # GPU
    #
    # for bz in range(batch_size):
    #     target_flow[bz, 0, :, :] = fmap[bz, 0, :, :]
    #     target_flow[bz, 1, :, :] = fmap[bz, 1, :, :]

    EPE_map = torch.norm(target_flow-input_flow,2,1)

    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

def evaluate(input_flow, fmap):

    pred = 20*input_flow
    # calculate end point error
    lable_Xb = 20*fmap[:, 0, :, :]
    lable_Yb = 20*fmap[:, 1, :, :]
    predict_Xb = pred[:, 0, :, :]
    predict_Yb = pred[:, 1, :, :]
    flag = fmap[:, 2, :, :]

    # Euclidean distance
    error_matrix = ((predict_Xb - lable_Xb)**2 + (predict_Yb - lable_Yb)**2)**0.5
    EffectivePoints_error_matrix = error_matrix * flag

    errorsum = torch.sum(EffectivePoints_error_matrix)
    count = torch.sum(flag)
    if count == 0:
        return False

    EndPointError = errorsum / count
    return EndPointError

def ACE(input_flow, fmap):
    pred = 20 * input_flow
    # calculate end point error
    lable_Xb = 20 * fmap[:, 0, :, :]
    lable_Yb = 20 * fmap[:, 1, :, :]
    predict_Xb = pred[:, 0, :, :]
    predict_Yb = pred[:, 1, :, :]

    predict_Xb = predict_Xb.numpy()[0]
    predict_Yb = predict_Yb.numpy()[0]
    lable_Xb   = lable_Xb.numpy()[0]
    lable_Yb   = lable_Yb.numpy()[0]
    ##########################################################################
    # calculate ACE
    rho = 32
    patch_size = 128
    top_x = random.randint(0 + rho, 320 - rho - patch_size)
    top_y = random.randint(0 + rho, 240 - rho - patch_size)
    # top_left_point = (top_x, top_y)
    # top_right_point = (top_x + patch_size, top_y)
    # bottom_left_point = (top_x, top_y + patch_size)
    # bottom_right_point = (top_x + patch_size, top_y + patch_size)
    # four_points = [top_left_point, top_right_point, bottom_left_point, bottom_right_point]

    predict_top_left_point = (predict_Xb[top_x, top_y], predict_Yb[top_x, top_y])
    predict_top_right_point = (predict_Xb[top_x + patch_size, top_y], predict_Yb[top_x + patch_size, top_y])
    predict_bottom_left_point = (predict_Xb[top_x, top_y + patch_size], predict_Yb[top_x, top_y + patch_size])
    predict_bottom_right_point = (predict_Xb[top_x + patch_size, top_y + patch_size], predict_Yb[top_x + patch_size, top_y + patch_size])

    lable_top_left_point = (lable_Xb[top_x, top_y], lable_Yb[top_x, top_y])
    lable_top_right_point = (lable_Xb[top_x + patch_size, top_y], lable_Yb[top_x + patch_size, top_y])
    lable_bottom_left_point = (lable_Xb[top_x, top_y + patch_size], lable_Yb[top_x, top_y + patch_size])
    lable_bottom_right_point = (lable_Xb[top_x + patch_size, top_y + patch_size]), lable_Yb[top_x + patch_size, top_y + patch_size]

    predict_four_points = [predict_top_left_point, predict_top_right_point, predict_bottom_left_point,predict_bottom_right_point]
    lable_four_points = [lable_top_left_point, lable_top_right_point, lable_bottom_left_point, lable_bottom_right_point]

    Avg_Corner_Error = np.sum(np.subtract(np.array(predict_four_points), np.array(lable_four_points)) ** 2) / 8

    return Avg_Corner_Error
    ##########################################################################


def realimageEPE(output, target):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return EPE_Loss(upsampled_output, target, mean=True)

