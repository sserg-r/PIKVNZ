class COBresolve_image:
    import tensorflow as tf
    import os
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   

    
    def __init__(self, COBmodel):
        self.model=COBmodel
#         assert os.path.isfile(self.path_to_image), '{0} - not valid file path'.format(self.path_to_image)

    def preprocess_image(self, path_to_image):
        import numpy as np
        from PIL import Image
        img = np.asarray(Image.open(path_to_image)) 
        mn=np.array([122.67891434, 116.66876762,104.00698793])
        data=(img[:,:,:3]-mn)[:,:,[2,1,0]]
        return np.transpose(data,axes=(1,0,2))
        
    def get_l_p(self, L):
        import math
        '''calculate division parameters of line of size L to overlapping segments,
        return dict of {l:segment length, p: overlapping, n: segment counts, res: overage of L}'''

        if L<=500: return {'l': L, 'p': 0, 'n': 1, 'res': 0}    
    #     assert L>500, 'L must be greater then 500'
        p0=100
        l0=500
        n=math.ceil((L-p0)/(l0-p0))
        L_hat=l0*n-(n-1)*p0
        dL=L_hat-L    
        diff=dL
        a0=0
        b0=0
        for a in range (dL//n,-1,-1):
            dl_res=dL%(n*a) if a>0 else dL
            b=dl_res//(n-1)

            if dl_res%(n-1)==0:
                a0=a
                b0=b
                diff=0
                break        

            if dl_res%(n-1)<diff:        
                diff=dl_res%(n-1)
                a0=a
                b0=b               

        return {'l': l0-a0,'p':b0+p0, 'n': n, 'res':diff}

    def batcherize (self, image):
        import numpy as np
        '''divide a 3-dims image with the sum of 2 dimensions more than 1000 into n patches with the size of h or w no more than 500'''
        h,w,_=image.shape    
        out_batch=[]    
        if h+w<=1000:
            out_batch.append(image)
            return out_batch, (0,0), (1,1), (0,0)
        else:
            par_h=self.get_l_p(h)
            par_w=self.get_l_p(w)
            tim_padded=np.pad(image, ((0,par_h['res']),(0,par_w['res']),(0,0)),mode='symmetric')
            step_h=par_h['l']-par_h['p']
            step_w=par_w['l']-par_w['p']        
            for ih in range(par_h['n']):
                for iw in range( par_w['n']):  
                    out_batch.append(tim_padded[ih*step_h:ih*step_h+par_h['l'],iw*step_w:iw*step_w+par_w['l'],:])

        overlapping=(par_h['p'], par_w['p'])
        res=(par_h['res'], par_w['res'])
        n=(par_h['n'],par_w['n'])
        return out_batch, overlapping, n, res
    
    def interpolate_angles(self, I): 
        import numpy as np
        '''
        Interpolating orientations strength maps to 1 orientation map

        I - np array of orientation strength maps size of [n, h,w] n -numbers of maps

        '''

        numClusters=I.shape[0]
        [Max1,ind1] = [np.max(I,axis=0), np.argmax(I,axis=0)]

        for i in range(numClusters):
            I[i,:,:]=(I[i,:,:]!=Max1)*I[i,:,:]

        [Max2,ind2] = [np.max(I,axis=0), np.argmax(I,axis=0)]    

        ind1=ind1+1
        ind2=ind2+1
        ind1[(ind1*ind2>0)*(ind2-ind1==numClusters-1)]=numClusters+1
        ind2[(ind1*ind2>0)*(ind1-ind2==numClusters-1)]=numClusters+1
        ind2[(ind1*ind2>0)*abs(ind1-ind2)>1]=0

        ab=np.vstack((ind1[None,:],ind2[None,:]))
        w1w2=np.vstack((Max1[None,:],Max2[None,:]))

        ind1[Max1<0.01]=0
        ind2[Max2<0.01]=0

        O=(np.sum(ab*w1w2,axis=0)/np.sum((ab>0)*w1w2,axis=0)*((ind1*ind2==0)+(abs(ind1-ind2)==1))-1)*np.pi/numClusters

        O[Max1<0.5]=-1
        O[O<0] = np.pi*np.random.rand(np.sum(O<0))
        return O
    
    def merger (self, batch, overlapping, n, res):
        import numpy as np
    
        '''merges batch of images in 1 image taking into account number of patches in h & w dims, 
        image's overlapping and trim resulting image right and bottom according to the res'''


#         batch=np.transpose(batch, (0,2, 3, 1))
#         N, h,w,b=batch.shape
        N, b, h,w=batch.shape
        if N==1: return batch[0]
#         print(N, h,w,b)

        step_h=h-overlapping[0]
        step_w=w-overlapping[1]


        L_h=h*n[0]-(n[0]-1)*overlapping[0]
        L_w=w*n[1]-(n[1]-1)*overlapping[1]

#         out=np.zeros((L_h,L_w,b))
        out=np.zeros((b, L_h,L_w))
        ittr=iter(batch)

        for ih in range(n[0]):
            shift_h=overlapping[0]//2 if ih!=0 else 0 
            for iw in range( n[1]):
                shift_w=overlapping[1]//2 if iw!=0 else 0                         
                out[:,ih*step_h+shift_h:ih*step_h+h,iw*step_w+shift_w:iw*step_w+w]=next(ittr)[:,shift_h:,shift_w:]

        return out[:,:L_h-res[0],:L_w-res[1]]
    
    
    def resolve_image(self, path_to_image):
        import os.path 
        assert os.path.isfile(path_to_image), '{0} - not valid file path'.format(path_to_image)
        
        import numpy as np
        prepr_im=self.preprocess_image(path_to_image)
        batches, overlapping, n, res=self.batcherize (prepr_im)
        predictions=[]
        for i in batches:            
            raw_pred=[i[0,:,:,0] for i in self.model.predict(i[None])]
            
            angles=self.interpolate_angles(np.array(raw_pred[:-2]))
            predictions.append([raw_pred[-1],raw_pred[-2],angles])
        
        outp=self.merger (np.array(predictions), overlapping, n, res)
        return np.transpose(outp,axes=(0,2,1))
    
    def resolve_imar(self, image_array):        
        
        import numpy as np
        
        mn=np.array([122.67891434, 116.66876762,104.00698793])
        data=(image_array[:,:,:3]-mn)[:,:,[2,1,0]]
        prepr_im= np.transpose(data,axes=(1,0,2))        
        batches, overlapping, n, res=self.batcherize (prepr_im)
        predictions=[]
        for i in batches:            
            raw_pred=[i[0,:,:,0] for i in self.model.predict(i[None])]
            
            angles=self.interpolate_angles(np.array(raw_pred[:-2]))
            predictions.append([raw_pred[-1],raw_pred[-2],angles])
        
        outp=self.merger (np.array(predictions), overlapping, n, res)
        return np.transpose(outp,axes=(0,2,1))
    
        