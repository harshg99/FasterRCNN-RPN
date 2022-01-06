import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import BuildDataset, BuildDataLoader
from utils import *
import pdb
import torchvision


class RPNHead(torch.nn.Module):

    def __init__(self,  device='cuda', anchors_param=dict(ratio=1.04, scale=154, grid_size=(50, 68), stride=16)):
        # Initialize the backbone, intermediate layer clasifier and regressor heads of the RPN
        super(RPNHead,self).__init__()

        self.device = device

        # TODO Define Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, (5, 5), stride=1, padding="same", bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2),stride= 2,padding =0),
            nn.Conv2d(16, 32, (5, 5), stride=1, padding="same", bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2),stride= 2,padding =0),
            nn.Conv2d(32, 64, (5, 5), stride=1, padding="same", bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2),stride= 2,padding =0),
            nn.Conv2d(64, 128, (5, 5), stride=1, padding="same", bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2),stride= 2,padding =0),
            nn.Conv2d(128, 256, (5, 5), stride=1, padding="same", bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        # TODO  Define Intermediate Layer
        self.intermediate_layer = nn.Sequential(
            nn.Conv2d(256, 256, (3, 3), stride=1, padding="same", bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        # TODO  Define Proposal Classifier Head
        self.classifier_head = nn.Sequential(
            nn.Conv2d(256, 1, (1, 1), stride=1, padding="same", bias=False),
            nn.Sigmoid()
        )

        # TODO Define Proposal Regressor Head
        self.proposal_head = nn.Sequential(
            nn.Conv2d(256, 4, (1, 1), stride=1, padding="same", bias=False),
        ) 
        
        # find anchors
        self.anchors_param=anchors_param
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict={}





    # Forward  the input through the backbone the intermediate layer and the RPN heads
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       logits: (bz,1,grid_size[0],grid_size[1])}
    #       bbox_regs: (bz,4, grid_size[0],grid_size[1])}
    def forward(self, X):

        #TODO forward through the Backbone
        X = self.forward_backbone(X)

        #TODO forward through the Intermediate layer
        X = self.intermediate_layer(X)

        #TODO forward through the Classifier Head
        logits = self.classifier_head(X)

        #TODO forward through the Regressor Head
        bbox_regs = self.proposal_head(X)
        
        assert logits.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert bbox_regs.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return logits, bbox_regs




    # Forward input batch through the backbone
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       X: (bz,256,grid_size[0],grid_size[1])
    def forward_backbone(self,X):
        #####################################
        # forward through the backbone
        #####################################
        X = self.backbone(X)
        assert X.shape[1:4]==(256,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return X



    # This function creates the anchor boxes
    # Output:
    #       anchors: (grid_size[0],grid_size[1],4)
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        ######################################
        # create anchors
        ######################################

        anchors = np.zeros((grid_sizes[0], grid_sizes[1], 4))
        h = scale / np.sqrt(aspect_ratio)
        w = aspect_ratio * h
        for i in range(grid_sizes[0]):
            for j in range(grid_sizes[1]):
                center_x = i * stride + (stride/2)
                center_y = j * stride + (stride/2)
                anchors[i, j, 0] = center_x - w/2
                anchors[i, j, 1] = center_y - h/2
                anchors[i, j, 2] = center_x + w/2
                anchors[i, j, 3] = center_y + h/2

        anchors[:,:,[0, 2]] = np.clip(anchors[:,:,[0, 2]], 0, grid_sizes[0] * stride)
        anchors[:,:,[1, 3]] = np.clip(anchors[:,:,[1, 3]], 0, grid_sizes[1] * stride)

        assert anchors.shape == (grid_sizes[0], grid_sizes[1], 4)

        return anchors


    def get_anchors(self):
        return self.anchors



    # This function creates the ground truth for a batch of images by using
    # create_ground_truth internally
    # Input:
    #      bboxes_list: list:len(bz){(n_obj,4)}
    #      indexes:      list:len(bz)
    #      image_shape:  tuple:len(2)
    # Output:
    #      ground_clas: (bz,1,grid_size[0],grid_size[1])
    #      ground_coord: (bz,4,grid_size[0],grid_size[1])
    def create_batch_truth(self,bboxes_list,indexes,image_shape):
        #####################################
        # TODO create ground truth for a batch of images
        #####################################
        bz = len(indexes)
        ground_clas = torch.zeros((bz, 1, self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1]))
        ground_coord = torch.zeros((bz, 4, self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1]))

        for i, (bboxes,index) in enumerate(zip(bboxes_list,indexes)):
            gcls, gcoord = self.create_ground_truth(bboxes, index, self.anchors_param['grid_size'], self.anchors, image_shape)
            ground_clas[i] = gcls
            ground_coord[i] = gcoord

        assert ground_clas.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return ground_clas, ground_coord


    # This function creates the ground truth for one image
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset used for caching)
    #       grid_size:   tuple:len(2)
    #       anchors:     (grid_size[0],grid_size[1],4)
    #       image_size:  tuple:len(2)
    # Output:
    #       ground_clas:  (1,grid_size[0],grid_size[1])
    #       ground_coord: (4,grid_size[0],grid_size[1])
   
    def create_ground_truth(self, bboxes, index, grid_size, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        #####################################################
        # create ground truth for a single image
        #####################################################

        ground_clas = torch.zeros((1,grid_size[0],grid_size[1])) - 1 
        ground_coord = torch.zeros((4,grid_size[0],grid_size[1]))
        
        iou_scores = IOU_vectorized(bboxes, anchors)
        self.image_size = image_size
        ground_clas[:,torch.all(iou_scores<0.3,dim=2)] = 0
        for i in range(iou_scores.shape[2]):
            ground_clas[:,iou_scores[:,:,i]==torch.max(iou_scores[:,:,i])] = 1
            ground_clas[:,iou_scores[:,:,i]>0.7] = 1
            for j in range(iou_scores.shape[0]):
                for k in range(iou_scores.shape[1]):
                    if(ground_clas[0,j,k]==1):
                        ground_coord[0,j,k] = ((bboxes[i][0] + bboxes[i][2])/2 - (anchors[j,k][0] + anchors[j,k][2])/2)/ np.abs(anchors[j,k][0] - anchors[j,k][2])
                        ground_coord[1,j,k] = ((bboxes[i][1] + bboxes[i][3])/2 - (anchors[j,k][1] + anchors[j,k][3])/2)/ np.abs(anchors[j,k][1] - anchors[j,k][3])
                        ground_coord[2,j,k] = np.log(np.abs(bboxes[i][0] - bboxes[i][2])/ np.abs(anchors[j,k][0] - anchors[j,k][2]))
                        ground_coord[3,j,k] = np.log(np.abs(bboxes[i][1] - bboxes[i][3])/ np.abs(anchors[j,k][1] - anchors[j,k][3]))
        self.ground_dict[key] = (ground_clas, ground_coord)

        assert ground_clas.shape==(1,grid_size[0],grid_size[1])
        assert ground_coord.shape==(4,grid_size[0],grid_size[1])
        return ground_clas, ground_coord





    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self,p_out,n_out):

        #torch.nn.BCELoss()
        # compute classifier's loss

        loss = torch.sum(-torch.log(p_out))+torch.sum(-torch.log(1 - n_out))
        sum_count = p_out.shape[0] + n_out.shape[0]
        return loss,sum_count



    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self,pos_target_coord,pos_out_r):
        #torch.nn.SmoothL1Loss()
        # compute regressor's loss

        sum_count = pos_target_coord.shape[0]
        loss_func = torch.nn.SmoothL1Loss()
        loss = torch.sum(loss_func(pos_target_coord, pos_out_r))
        return loss, sum_count



    # Copute the total loss
    # Input:
    #       clas_out: (bz,1,grid_size[0],grid_size[1])
    #       regr_out: (bz,4,grid_size[0],grid_size[1])
    #       targ_clas:(bz,1,grid_size[0],grid_size[1])
    #       targ_regr:(bz,4,grid_size[0],grid_size[1])
    #       l: lambda constant to weight between the two losses
    #       effective_batch: the number of anchors in the effective batch (M in the handout)
    def compute_loss(self,clas_out,regr_out,targ_clas,targ_regr, l=2, effective_batch=100):
        #############################
        # compute the total loss
        #############################
        regr_out_flat, clas_out_flat,_ = output_flattening(regr_out, clas_out, self.anchors)
        targ_regr_flat, targ_clas_flat,_ = output_flattening(targ_regr, targ_clas, self.anchors)
        clas_out_pos =[]
        regr_out_pos =[]
        clas_out_neg = []
        targ_regr_pos = []

        if clas_out_flat[targ_clas_flat>0].shape[0] >= (effective_batch / 2):
            idx = np.random.choice(int(clas_out_flat[targ_clas_flat>0].shape[0]), int(effective_batch/2),replace=False)
            clas_out_pos = clas_out_flat[targ_clas_flat>0][idx]
            clas_out_neg = clas_out_flat[targ_clas_flat==0][np.random.choice(int(clas_out_flat[targ_clas_flat==0].shape[0]), int(effective_batch/2),replace=False)]
            regr_out_pos = regr_out_flat[(targ_clas_flat>0).squeeze(),][idx]
            targ_regr_pos = targ_regr_flat[(targ_clas_flat>0).squeeze(),][idx]
        else:
            clas_out_pos = clas_out_flat[targ_clas_flat>0]
            regr_out_pos = regr_out_flat[(targ_clas_flat>0).squeeze(),]
            clas_out_neg = clas_out_flat[targ_clas_flat==0][np.random.choice(int(clas_out_flat[targ_clas_flat==0].shape[0]), int(effective_batch - clas_out_flat[targ_clas_flat>0].shape[0]))]
            targ_regr_pos = targ_regr_flat[(targ_clas_flat>0).squeeze(),]
        
        loss_c,sum_count1 = self.loss_class(clas_out_pos,clas_out_neg)
        loss_r,sum_count2 = self.loss_reg(targ_regr_pos,regr_out_pos)
        loss = loss_c/sum_count1 + l*loss_r
        return loss, loss_c/sum_count1, loss_r



    # Post process for the outputs for a batch of images
    # Input:
    #       out_c:  (bz,1,grid_size[0],grid_size[1])}
    #       out_r:  (bz,4,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinates of the boxes that the NMS kept)
    def postprocess(self,out_c,out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
        nms_clas_list = []
        nms_prebox_list =[]
        pre_nms_clas_list = []
        pre_nms_prebox_list =[]
        print(out_c.shape[0])
        for i in range(out_c.shape[0]):
            batch_nms_score_val = self.postprocessImg(out_c[i, :, :, :], out_r[i,:, :, :], IOU_thresh, keep_num_preNMS, keep_num_postNMS)
            #here I am passing one image to the postprocessImg function 
            nms_clas_list.append(batch_nms_score_val[0])
            nms_prebox_list.append(batch_nms_score_val[1])
            pre_nms_clas_list.append(batch_nms_score_val[2])
            pre_nms_prebox_list.append(batch_nms_score_val[3])
        return nms_clas_list, nms_prebox_list,pre_nms_clas_list, pre_nms_prebox_list



    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)
    def postprocessImg(self,mat_clas,mat_coord, IOU_thresh,keep_num_preNMS, keep_num_postNMS):
            ######################################
            # TODO postprocess a single image
            #####################################
            
        #Clipping
        # set all boxes whose x1 y1 <0  to 0
        # set all boxes whose x2 y2 >0 800,1088 to 800 ,1088
        flat_out_r,flat_out_c,flat_anchors = output_flattening(mat_coord.unsqueeze(0),mat_clas.unsqueeze(0),self.get_anchors())
        bbox_decoded = output_decoding(flat_out_r,flat_anchors)
        bbox_decoded[bbox_decoded[:,0]<0,0] = 0
        bbox_decoded[bbox_decoded[:,1]<0,1] = 0
        bbox_decoded[bbox_decoded[:,2]>self.image_size[0],2]= self.image_size[0]
        bbox_decoded[bbox_decoded[:,3]>self.image_size[1],3] = self.image_size[1]
        
        # Top 50 proposals by doing max of out_c
        top_proposals =[]
        sort_clas,indices = torch.sort(flat_out_c,dim = 0, descending=True)
        sort_reg = bbox_decoded[indices,:]

        top_proposals_indices = indices[:keep_num_preNMS]
        top_proposals_clas = sort_clas[:keep_num_preNMS]
        top_proposals_reg = sort_reg[:keep_num_preNMS,:]

        top_proposals_after_NMS = []

        nms_clas,nms_prebox = self.NMS(top_proposals_clas.clone(), top_proposals_reg.clone(),IOU_thresh)  
        #passing one grid_size value to the NMS function

        nms_clas = nms_clas[:keep_num_postNMS]
        nms_prebox = nms_prebox[:keep_num_postNMS,:]
        return nms_clas, nms_prebox,top_proposals_clas,top_proposals_reg


    # Input: 
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self,clas,prebox, thresh):
        ##################################
        # TODO perform Nms
        ##################################
        nms_clas = []
        nms_prebox = []
        clas_copy = torch.clone(clas).squeeze().detach().cpu().numpy()
        while True:
            if(np.count_nonzero(clas_copy)==0):
                break
            current_box = np.argwhere(clas_copy==clas_copy.max())
            box = torch.clone(prebox[current_box,:].squeeze())
            clas_copy[current_box] = 0
            iou = IOU(box,prebox.squeeze())
            supp = torch.logical_and(iou>thresh,torch.abs(iou-1.0)>0.0001)
            # supresses all boxes that are above 0.5 iou
            clas_copy[supp] = 0
            clas[supp] = 0
        return clas[clas>0],prebox[clas>0]
    
if __name__=="__main__":
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

  
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    torch.random.manual_seed(1)    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = iter(train_build_loader.loader())
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = iter(test_build_loader.loader())

    #I AM COMMENTING THE CODE HERE, FOR NOW#    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    rpn_net = RPNHead()
    # push the randomized training data into the dataloader

    
    for i,batch in enumerate(train_loader,0):
        images=batch['images'][0,:,:,:]
        indexes=batch['index']
        boxes=batch['bbox']
        gt,ground_coord=rpn_net.create_batch_truth(boxes,indexes,images.shape[-2:])

        # Flatten the ground truth and the anchors
        flatten_coord,flatten_gt,flatten_anchors=output_flattening(ground_coord,gt,rpn_net.get_anchors())
        
        # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
        decoded_coord=output_decoding(flatten_coord,flatten_anchors)
        
        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        images = transforms.functional.normalize(images,
                                                      [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                                      [1/0.229, 1/0.224, 1/0.225], inplace=False)

        pred_class,pred_coord = rpn_net(images)
        rpn_net.compute_loss(pred_class,pred_coord,gt,ground_coord)

        fig,ax=plt.subplots(1,1)
        ax.imshow(images.permute(1,2,0))
        
        find_cor=(flatten_gt==1).nonzero()
        find_neg=(flatten_gt==-1).nonzero()
             
        for elem in find_cor:
            coord=decoded_coord[elem,:].view(-1)
            anchor=flatten_anchors[elem,:].view(-1)

            col='r'
            rect=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
            ax.add_patch(rect)
            rect=patches.Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
            ax.add_patch(rect)

        plt.show()
 
        if(i>20):
            break