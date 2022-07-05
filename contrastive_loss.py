import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

class Contrast_loss_point_cloud(nn.Module):
        def __init__(self, temperature=0.7):
            super(Contrast_loss_point_cloud, self).__init__()
            self.temp = temperature
            self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        def forward(self, features, labels_all=None):
            all_loss = []
            for features_map,labels in zip(features,labels_all):


                labels = labels.unsqueeze(0)


                normalize_vectors = F.normalize(features_map.T,p = 2,dim = 1)
                # norms  = torch.matmul(torch.norm(normalize_vectors, dim=1).unsqueeze(1) , torch.norm(normalize_vectors, dim=1).unsqueeze(1).T)       
                dot_products = torch.matmul(normalize_vectors, normalize_vectors.T) 
                # dot_products = torch.div(dot_products,norms)
                dot_products = torch.div(dot_products,self.temp)
                dot_products = torch.exp(dot_products)
                
                dot_products = dot_products - torch.diag(torch.diagonal(dot_products, 0))

                mask = torch.eq(labels, labels.T).float()
                # mask_not = torch.logical_not(mask)


                posetives = (mask * dot_products).sum(1) 
                negetives = (dot_products).sum(1)
                # print(posetives,negetives)

                diviation = posetives / (negetives)
                # print(diviation)
                
                diviation = - torch.log(diviation)
                # print(diviation)
                loss = torch.mean(diviation)
                # print(loss)
                # print("------------------------------------------")
                if torch.isinf(loss) == False and torch.isnan(loss) == False:
                    # print("loss=>",loss)
                    all_loss.append(loss)
                else:
                    print("inf or nan loss founded")
            all_loss = torch.stack(all_loss)
            # print(all_loss)
            return torch.sum(all_loss)



class Contrast_loss_point_cloud_inetra_batch(nn.Module):
        def __init__(self, temperature=0.07):
            super(Contrast_loss_point_cloud_inetra_batch, self).__init__()
            self.temp = temperature
            self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        def forward(self, features_in, labels_in=None):
            t1 = time()
            # all_loss = []
            # for features_map,labels in zip(features,labels_all):
            features_shape = features_in.shape
            features= features_in.view(features_shape[1],features_shape[0]*features_shape[2])
            t2 = time()
            ############
            labels = labels_in.flatten()
            t3 = time()
            dist = 400 / (torch.bincount(labels) +1)
            stats = torch.empty(labels.shape)
            stats = labels
            stats = stats.double()
            for i in range(dist.shape[0]):
                if dist[i] > 1:
                  dist[i] = 1
                stats[stats == i] = dist[i].item()
    

            t4 = time()
            # for i in range(labels.shape[0]):
            #     stats[i] = dist[labels[i]]
            t5 = time()
            mask_label = torch.bernoulli(stats).to(self.device)
            mask_select = mask_label > 0 
            labels = torch.masked_select(labels, mask_select)
            mask_data = torch.nonzero(mask_select).flatten()
            t6 = time()
            features = features.T[mask_data,:]
            ###############
            # print("labels",torch.bincount(labels))
            t7 = time()
            labels = labels.unsqueeze(0)
            normalize_vectors = F.normalize(features,p = 2,dim = 1)
            norms  = torch.matmul(torch.norm(normalize_vectors, dim=1).unsqueeze(1) , torch.norm(normalize_vectors, dim=1).unsqueeze(1).T)       
            dot_products = torch.matmul(normalize_vectors, normalize_vectors.T) 
            dot_products = torch.div(dot_products,norms)

            dot_products = torch.div(dot_products,self.temp)
            dot_products = torch.exp(dot_products)
            
            dot_products = dot_products - torch.diag(torch.diagonal(dot_products, 0))

            mask = torch.eq(labels, labels.T).float()

            mask_not = torch.logical_not(mask)


            posetives = (mask * dot_products).sum(1) 
            negetives = (dot_products).sum(1)

                
            diviation = posetives / (negetives)

            #     # print(diviation)
                
            diviation = - torch.log(diviation)
            #     # print(diviation)
            loss = torch.mean(diviation)
            t8 = time()
            # print("time",t2 - t1 , t3 - t2,t4 - t3,t5 - t4,t6 - t5,t7 - t6,t8 - t7)
            #     # print(loss)
            #     # print("------------------------------------------")
            if torch.isinf(loss) == False and torch.isnan(loss) == False:

                    return torch.sum(loss)
            else:
                    print("inf or nan loss founded")
                    loss = 0
                    return torch.torch(0)





if __name__ == '__main__':    
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    feature = torch.rand(2,128,4096).to(device)
    labels = torch.randint(0, 10, (3,4096)).to(device)
    loss_class = Contrast_loss_point_cloud().to(device)
    loss_class(feature,labels)








