import rasterio
from rasterio import features
import fiona
from shapely.geometry import shape
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

class Segmentation:
    def __init__(self, line_thresh=0.1, area_thresh=10):
        self.line_thresh=line_thresh,
        self.area_thresh=area_thresh
    def get_segments_table(self,segment_map, image):
        '''
        segment_map - indexed connected pixels groups where 0 - labeling masked area
        '''
        from skimage import measure
        from scipy import ndimage
        import numpy as np
        #     import pandas as pd    
        h,w,d=image.shape    
        index, reg_area = np.unique(segment_map,return_counts=True)    

        coord_lists=[(segment_map.ravel()==ind).nonzero()[0] for ind in index] 

        region_means=[np.array(i) for i in zip(*[ndimage.mean(image[:,:,band], labels=segment_map, index=index) for band in range(d)])]    
        return {index[i]: [reg_area[i], region_means[i]] for i in range (len(index))}

    def get_board(self,segmented):
        "return boarder's length" 
        from scipy.signal import correlate2d
        import numpy as np
        kernel=[[0,1,0],[1,0,1],[0,1,0]]
        outlist={}
        for idx in np.unique(segmented):    
            corr=correlate2d((segmented==idx),kernel,mode='same')
            bord=corr*(segmented!=idx)        
            outlist[idx]= [(zone,np.sum(bord[(segmented==zone) & (bord>0)])) for zone in np.unique(segmented[bord>0])]
        return outlist

    def get_dist(self,fv1,fv2):
        import numpy as np
        return np.sum((fv1-fv2)**2)

    def get_neighbs_table(self,hhh, all_labels, thresh=100, zero_is_mask=True):
        import numpy as np
        thresh=thresh
        jj= self.get_board(all_labels)
        outdict={}
        for key in jj:
            if key==0 and zero_is_mask:
                continue
            for kk, lent in jj[key]:
                if kk==0 and zero_is_mask:
                    continue
                areas=np.array([hhh[key][0],hhh[kk][0]])     

                if kk>key and np.min(areas)<thresh:
                    features= np.array([hhh[key][1],hhh[kk][1]]) 
                    dist=self.get_dist(*features)
                    metr=(np.sum(areas)/np.prod(areas))*dist/lent
                    outdict[(key, kk)]=[metr,lent]
        return outdict

    def clear_merger (self,merge_list):
        merger=[[],[]]
        for i in merge_list[::-1]:
            if i[0] in merger[1]:            
                merger[0].append(merger[0][merger[1].index(i[0])])
                merger[1].append(i[1])
            else: 
                merger[0].append(i[0])
                merger[1].append(i[1])
        return merger

    def get_clear_segm(self,all_labels, testim, thresh=100, zero_is_mask=True):
        import numpy as np
        hhh=self.get_segments_table(all_labels, testim)
        outdict=self.get_neighbs_table(hhh, all_labels, thresh, zero_is_mask)
        merger=[]

        while len(outdict)>0:
            minind=np.argmin(np.array(list(outdict.values()))[:,0])
            minpair=list(outdict.keys())[minind]
            merger.append(minpair)

            p,q=minpair
            ap, fp=hhh[p]
            aq, fq=hhh[q]
            hhh[p]=[ap+aq, (ap*fp+aq*fq)/(ap+aq)]


            for pair in [pair for pair in outdict if q in pair]:    
                pair_new = tuple([p if x==q else x for x in pair])
                if pair_new[0]==pair_new[1]: # выкинули старую пару
                    outdict.pop(pair)
                    continue    
                elif outdict.get(pair_new): # если пара с таким индексом уже существует
                    lent=outdict[pair_new][1]+outdict[pair][1]        
                else: # если такого индекса не существует
                    lent=outdict[pair][1]
                    outdict.pop(pair)

                key,kk=pair_new

                features= np.array([hhh[key][1],hhh[kk][1]]) 
                dist=self.get_dist(*features)
                areas=np.array([hhh[key][0],hhh[kk][0]])
                metr=(np.sum(areas)/np.prod(areas))*dist/lent
                outdict[(key, kk)]=[metr,lent] 


            for key,kk in list(outdict.keys()):
                areas=np.array([hhh[key][0],hhh[kk][0]])
                if np.min(areas)>thresh:
                    outdict.pop((key,kk)) 

        all_label1=np.copy(all_labels)

        for i in zip(*self.clear_merger(merger)):        
            all_label1[all_label1==i[1]]=i[0] 
        return all_label1


    def get_contour_masked(self,output, masked):
        import numpy as np
        import higra as hg
        hh=np.copy(masked)
        hh[hh>0]=255

        from scipy import signal

        Ix=signal.correlate2d(hh[:,:,0],[[1,2,1],[0,0,0],[-1,-2,-1]], mode='same',boundary='symm')
        Iy=signal.correlate2d(hh[:,:,0],[[1,0,-1],[2,0,-2],[1,0,-1]], mode='same',boundary='symm')

        G = np.hypot(Ix, Iy)

        G = G / G.max()

        size = hh.shape[:2]

        gradient_coarse = np.array([output[1],G]).max(axis=0)
        gradient_fine = np.array([output[0],G]).max(axis=0)
        gradient_orientation = output[2]


        graph = hg.get_4_adjacency_graph(size)
        edge_weights_fine = hg.weight_graph(graph, gradient_fine, hg.WeightFunction.mean)
        edge_weights_coarse = hg.weight_graph(graph, gradient_coarse, hg.WeightFunction.mean)
        edge_weights_hig = hg.weight_graph(graph, G, hg.WeightFunction.mean)


        # special handling for angles to wrap around the trigonometric cycle...
        edge_orientations_source = hg.weight_graph(graph, gradient_orientation, hg.WeightFunction.source) 
        edge_orientations_target = hg.weight_graph(graph, gradient_orientation, hg.WeightFunction.target) 
        edge_orientations = hg.mean_angle_mod_pi(edge_orientations_source, edge_orientations_target)

        combined_hierarchy1, altitudes_combined1 = hg.multiscale_mean_pb_hierarchy(graph, edge_weights_fine, others_edge_weights=(edge_weights_coarse,), edge_orientations=edge_orientations)

        return hg.graph_4_adjacency_2_khalimsky(graph, hg.saliency(combined_hierarchy1, altitudes_combined1))

    def get_segm_map(self,predicter, image, masked, line_thresh,area_thresh): 
        from skimage import measure
        import numpy as np
        line_thresh=line_thresh
        area_thresh=area_thresh
        im=masked
        mn= np.sum(im,axis=(0,1))/np.sum(im>0,axis=(0,1))
        mar=np.ma.array(im, mask=im>0)
        im_msk_mn=np.array(mar+mn.astype('uint8'))
        ott=np.array([image,im_msk_mn]).mean(axis=0).astype('uint8')
        cont_min_inside=self.get_contour_masked(predicter.resolve_imar(ott),im)
        outl=1-(cont_min_inside>line_thresh)
        segments= measure.label(outl)[::2,::2]

        if masked.ndim==3:
            masked=masked.prod(axis=2)>0

        segments[masked==0]=0
        segments= measure.label(segments)
        return self.get_clear_segm(segments,image,area_thresh)
    
    def segm_map_generator(self,list_of_param_dict):
        from rasterio import features
        from shapely.geometry import shape
        import numpy as np        
        from model import COBresolve_image
        from model.COB_model import get_COB_model
        import os
        weights_dir = 'weights'
        weights_name='COB_PASCALContext_trainval.h5'
        path_file = lambda dirr, filename: os.path.join(dirr, filename)
        input_shape=(None,None,3)
        model=get_COB_model(path_file(weights_dir,weights_name),input_shape)
        predicter=COBresolve_image(model)
              
        line_thresh=self.line_thresh
        area_thresh=self.area_thresh 
        for param_dict in list_of_param_dict:              
            image=param_dict['rgb_5']
            transform=param_dict['transf_5']
            geojson = param_dict['geojson']
            geom=shape(geojson['geometry'])
            rows, cols, _ = image.shape
            mask = features.rasterize([geom], out_shape=(rows, cols),transform=transform)
            masked=np.copy(image)
            masked[mask==0]=0
            param_dict['segments_5']=self.get_segm_map(predicter, image, masked, line_thresh, area_thresh)
            yield param_dict
            
            
            
def patch_generator(param_dict):
    '''accepts file path variables: "tif_10m", "tif_20m", "vectors", "s1tif_20m"'''
    dataset_10=rasterio.open(param_dict['tif_10m'])
    dataset_20=rasterio.open(param_dict['tif_20m'])
    datasets1_20=rasterio.open(param_dict['s1tif_20m']) 
    bands_10=dataset_10.descriptions
    bands_20=dataset_20.descriptions
    shp = fiona.open(param_dict['vectors'], layer=0)
    vect_meta=shp.meta
    
    
    for geojson in iter(shp):
        geom=shape(geojson['geometry'])
        if geom.area<100: continue
        wnd_10=features.geometry_window(dataset_10,[geojson],pad_x=6,pad_y=6, north_up=True, rotated=False, pixel_precision=3)
        wnd_20=features.geometry_window(dataset_20,[geojson],pad_x=3,pad_y=3, north_up=True, rotated=False, pixel_precision=3)
        wnds1_20=features.geometry_window(datasets1_20,[geojson],pad_x=3,pad_y=3, north_up=True, rotated=False, pixel_precision=3)
        
        
        offs_10=dataset_10.transform*(wnd_10.col_off,wnd_10.row_off)
        offs_20=dataset_20.transform*(wnd_20.col_off,wnd_20.row_off)

        
        
        transf_10=list(dataset_10.transform)[:-3]
        transf_20=list(dataset_20.transform)[:-3]

        transf_10[2]=offs_10[0]
        transf_10[5]=offs_10[1]
        
        transf_20[2]=offs_20[0]
        transf_20[5]=offs_20[1]
        
        
                
#         transf_10=dataset_10.transform.translation(*offs_10)
#         transf_20=dataset_20.transform.translation(*offs_20)        
        
        patch_10=dataset_10.read(window=wnd_10)
        patch_20=dataset_20.read(window=wnd_20)
        patchs1_20=datasets1_20.read(window=wnds1_20).squeeze()
        if 'B4'in bands_10 and 'B3' in bands_10 and 'B2' in bands_10:
            rgb_bands=[bands_10.index(bnum) for bnum in ['B4','B3','B2']]
        else: rgb_bands=[2,1,0]
        rgb=(np.moveaxis(patch_10,0,2))[:,:,rgb_bands]
        rgb=np.ceil(rgb/2500*255).astype('uint8')        
    
        yield {'patch_10':patch_10,'patch_20':patch_20,'patchs1_20':patchs1_20,'rgb_10':rgb, 'geojson':geojson,'transf_10':transf_10,'transf_20':transf_20, 'crs':dataset_10.crs, 'vect_meta':vect_meta}
        

def supres_generator(list_of_param_dict):
    import tensorflow as tf
    import os
    import numpy as np
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'    
    
    
    from model import generator
    from model import resolve_single
    from skimage.transform import rescale
    from skimage.filters import unsharp_mask

    weights_dir = 'weights'
    path_file = lambda dirr, filename: os.path.join(dirr, filename)

    pre_generator = generator()
    gan_generator = generator()
    pre_generator.load_weights(path_file(weights_dir,'pre_generator.h5'))
    gan_generator.load_weights(path_file(weights_dir,'gan_generator.h5'))
    
    for param_dict in list_of_param_dict:
        rgb5=(rescale(unsharp_mask(resolve_single(pre_generator, param_dict['rgb_10']), radius=3, amount=1,multichannel=True,), 0.5,multichannel=True,anti_aliasing=True)*255).astype('uint8')
        rgb5=np.clip(rgb5,1,255)
        transf_rgb5=list(param_dict['transf_10'])
        transf_rgb5[0]*=0.5
        transf_rgb5[4]*=0.5
        param_dict['rgb_5']=rgb5
        param_dict['transf_5']=transf_rgb5
        yield param_dict          
        

        
        
def simple_rescale(im, scf=2):
    'rescale image'
    import numpy as np    
    row,col=im.shape[0],im.shape[1]
    return np.array([[im[int(r/scf)][int(c/scf)] for c in range(col*scf)] for r in range(row*scf)])

# classification section

def class_generator(list_of_param_dict, rf_model_path='./model/RF_model.pkl'):
    import pickle
    
    import numpy as np
    import scipy
    
    with open(rf_model_path, 'rb') as f:
        clf=pickle.load(f)
    
    
    for param_dict in list_of_param_dict:        
        segm=param_dict['segments_5']
        transf_10=param_dict['transf_10']
        transf_20=param_dict['transf_20']
        patch_10=np.rollaxis(param_dict['patch_10'],0,3)
        patch_20=np.rollaxis(param_dict['patch_20'],0,3)
        patchs1_20=param_dict['patchs1_20']


        x10,y10,_=patch_10.shape
        y_sh,x_sh=np.abs(np.array([transf_10,transf_20]).T.dot(np.array([-1,1]))[[2,5]]/10).astype(int)
        patch_20to10=simple_rescale(patch_20)[x_sh:x_sh+x10,y_sh:y_sh+y10,:]
        feat_image=simple_rescale(np.concatenate((patch_10, patch_20to10), axis=2))


        patchs1_20to10=simple_rescale(patchs1_20)[x_sh:x_sh+x10,y_sh:y_sh+y10]
        patchs1_5=simple_rescale(patchs1_20to10)    #sent1 5m patch

        remap={}
        for index in np.unique(segm.ravel()):
            if index==0: continue        
            nm_pix=len(segm[segm==index])//4
        #     nm_samples=np.ceil((np.log(nm_pix)/np.log(2))**1.6).astype(int) 
            nm_samples=nm_pix
            id_choises=np.random.choice(range(nm_pix),nm_samples,replace=False)    
    #         is_candidate=(patchs1_5[segm==index].mean()+0.71).astype(int)
            is_candidate=(patchs1_5[segm==index].mean()+0.51).astype(int)
        #     is_candidate=(patchs1_5[segm==index][id_choises].mean()+0.6).astype(int)
            if not is_candidate:
                remap[index]=0
                continue   

        #     featured_items=feat_image[segm==index][id_choises]
            featured_items=feat_image[segm==index]
            cl=scipy.stats.mode(sorted(clf.predict(featured_items))).mode[0]
            remap[index]=cl
        src, values = remap.keys(), remap.values()
        d_array = np.arange(segm.max() + 1)
        d_array[list(src)] = list(values)
        class_image=d_array[segm]     
        class_image[segm==0]=-1   
        param_dict['classes']=class_image
        yield param_dict


# vectorization section

def nearest_ind(ind_neig,ind_cand):
    import numpy as np
    class_generator
    Xn=ind_neig[:,0]
    Yn=ind_neig[:,1]
    Xc=ind_cand[:,0]
    Yc=ind_cand[:,1]
    Xn=np.stack([-np.ones_like(Xn),Xn])
    Xc=np.stack([Xc, np.ones_like(Xc)])
    Xdist=Xn.T.dot(Xc)    
    Yn=np.stack([-np.ones_like(Yn),Yn])
    Yc=np.stack([Yc, np.ones_like(Yc)])
    Ydist=Yn.T.dot(Yc) 
    return np.argmin((Xdist**2+Ydist**2),axis=0)

def expand_im(im, mask_val=-1):
    import numpy as np
    from scipy.ndimage import binary_erosion
    mask=np.copy(im)
    mask[mask!=mask_val]=1
    mask[mask==mask_val]=0
    neig_mask=binary_erosion(mask)-mask
    indarr=np.indices(mask.shape)

    XYn=np.rollaxis(indarr,0,3)[neig_mask==-1]
    XYc=np.rollaxis(indarr,0,3)[mask==0]

    gg=nearest_ind(XYn,XYc)
    ccl=np.copy(im)
    ccl[XYc[:,0],XYc[:,1]]=ccl[XYn[gg][:,0],XYn[gg][:,1]]
    return ccl   

def vectors_generator(list_of_param_dict):
    from shapely.geometry import shape
#     from scipy import ndimage
    import rasterio
    import shapely
    import copy
    import numpy as np
    for param_dict in list_of_param_dict:
        cl1=expand_im(param_dict['classes']) 
        
        inshp=dict(param_dict['geojson'])        
        outline=shape(inshp['geometry'])        
        
        trans=np.array(param_dict['transf_5']).reshape((2,3))

        mypoly=[]

        for vec in rasterio.features.shapes(cl1.astype('int16')):
            multpl=vec[0]['coordinates']
            for i in range(len(multpl)):
                poly=multpl[i]
                poly=[tuple(trans.dot([x,y,1])) for x,y in poly]
                vec[0]['coordinates'][i]=poly
            shp=shape(vec[0])        
            shp=shp.intersection(outline)

            if shp.geom_type in['MultiPolygon','GeometryCollection']:
                if len(shp)==0: continue
                shps=[pol for pol in shp]
            else: shps=[shp]
            for shp in shps:
                outshp=copy.deepcopy(dict(inshp))
                outshp['properties']['class']=int(vec[1])
#                 prop= copy.deepcopy(inshp['properties'])    
#                 prop['class']=int(vec[1])
                
                outshp['geometry']= shapely.geometry.mapping(shp)
#                 outshp['properties']=prop
                mypoly.append(outshp) 

        param_dict['classed_geojson']=mypoly
        yield param_dict


        
        
#write vektors to shp
def write_to_shape(vectorized_items, shp_path):
    import copy
    import fiona
    polygs=[item for dct in vectorized_items for item in dct['classed_geojson']]
    meta=vectorized_items[0]['vect_meta']
    meta['schema']['properties']['class']='int:10'
    with fiona.open(shp_path, 'w',**meta) as sink:
        for poly in polygs:    
            sink.write(poly)         
        
        
        
        
        
        
        

def load_image(path):
    return np.array(Image.open(path))


def plot_sample(lr, sr):
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        
        
def text_phantom(text, size):
   
    font = 'Gargi'
    
    pil_font = ImageFont.truetype(font + ".ttf", size=size[1]*2 // len(text), encoding="unic")
    text_width, text_height = pil_font.getsize(text)
   
    canvas = Image.new('RGB', [size[1],size[0]], (255, 255, 255))
    
    draw = ImageDraw.Draw(canvas)
    offset = ((size[1] - text_width) // 2,
              (size[0] - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)
   
    return (255 - np.asarray(canvas)) / 255.0