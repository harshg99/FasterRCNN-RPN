import numpy as np
import torch
from functools import partial
import pdb

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function computes the IOU between two set of boxes
def IOU(boxA, boxB):
    ##################################
    # computes the IOU between the boxA, boxB boxes
    ##################################
    top_right = torch.minimum(boxA[[2, 3]], boxB[:,[2, 3]])
    bot_left = torch.maximum(boxA[[0, 1]], boxB[:,[0, 1]])
    intersection = ((top_right - bot_left).clamp(min=0)).prod(dim=1)
    union = (boxA[2] - boxA[0])*(boxA[3] - boxA[1]) + (boxB[:,2] - boxB[:,0])*(boxB[:,3] - boxB[:,1]) - intersection + 1e-6
    iou = intersection/union
    return iou


def IOU_vectorized(bboxes, anchors):
    ##################################
    # computes the IOU between the bounding boxes, anchors
    ##################################
    boxA = bboxes.reshape((1, bboxes.shape[0], bboxes.shape[1]))
    boxB = anchors.reshape((anchors.shape[0] * anchors.shape[1], 1, anchors.shape[2]))

    x1 = boxA[:,:,0]
    y1 = boxA[:,:,1]
    x2 = boxA[:,:,2]
    y2 = boxA[:,:,3]
    
    x3 = boxB[:,:,0]
    y3 = boxB[:,:,1]
    x4 = boxB[:,:,2]
    y4 = boxB[:,:,3]

    x1_int = np.maximum(x1, x3)
    y1_int = np.maximum(y1, y3)
    x2_int = np.minimum(x2, x4)
    y2_int = np.minimum(y2, y4)
    int_area = np.clip(x2_int - x1_int,0,None) * np.clip(y2_int - y1_int,0,None)
    u_area = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - int_area
    with np.errstate(divide='ignore'):
        iou = int_area / u_area
        iou[int_area == 0] = 0
    iou = iou.reshape((anchors.shape[0], anchors.shape[1], bboxes.shape[0]))
    return torch.from_numpy(iou)



# This function flattens the output of the network and the corresponding anchors 
# in the sense that it concatenates  the outputs and the anchors from all the grid cells
# from all the images into 2D matrices
# Each row of the 2D matrices corresponds to a specific anchor/grid cell
# Input:
#       out_r: (bz,4,grid_size[0],grid_size[1])
#       out_c: (bz,1,grid_size[0],grid_size[1])
#       anchors: (grid_size[0],grid_size[1],4)
# Output:
#       flatten_regr: (bz*grid_size[0]*grid_size[1],4)
#       flatten_clas: (bz*grid_size[0]*grid_size[1])
#       flatten_anchors: (bz*grid_size[0]*grid_size[1],4)
def output_flattening(out_r,out_c,anchors):
    #######################################
    # flatten the output tensors and anchors
    #######################################
    bz, _, grid_0, grid_1 = out_r.shape
    flatten_regr = out_r.permute((0,2,3,1)).reshape((bz * grid_0 * grid_1, 4))
    flatten_clas = out_c.permute((0,2,3,1)).reshape((-1,1))
    anchors = np.repeat(np.expand_dims(anchors,axis = 0), bz, axis=0)
    flatten_anchors = anchors.reshape((-1,anchors.shape[3]))
    return flatten_regr, flatten_clas, flatten_anchors




# This function decodes the output that is given in the encoded format (defined in the handout)
# into box coordinates where it returns the upper left and lower right corner of the proposed box
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
def output_decoding(flatten_out,flatten_anchors, device='cpu'):
    #######################################
    # TODO decode the output
    #######################################
    box = torch.zeros(flatten_out.shape)
    w = torch.exp(flatten_out[:,2])*torch.abs(torch.tensor(flatten_anchors[:,0] - flatten_anchors[:,2]))
    h = torch.exp(flatten_out[:,3])*torch.abs(torch.tensor(flatten_anchors[:,1] - flatten_anchors[:,3]))
    x = flatten_out[:,0]*(torch.abs(torch.tensor(flatten_anchors[:,0] - flatten_anchors[:,2]))) + torch.tensor((flatten_anchors[:,0] + flatten_anchors[:,2])/2)
    y = flatten_out[:,1]*(torch.abs(torch.tensor(flatten_anchors[:,1] - flatten_anchors[:,3]))) + torch.tensor((flatten_anchors[:,1] + flatten_anchors[:,3])/2)
    box[:,0] = x - w/2
    box[:,1] = y - h/2
    box[:,2] = x + w/2
    box[:,3] = y + h/2
    return box