import tensorflow as tf
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda, MaxPool2D, ReLU, Concatenate, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import layers


#Res-Net50 (backbone)

def res_block_simple(inp, stage, substage):    
    filters=inp.shape.as_list()[-1]//4
    conv=Conv2D(filters, 1, activation='relu', use_bias=False, name='res{0}{1}_branch2a'.format(stage, substage))(inp)   
    conv=Conv2D(filters, 3, padding='same', activation='relu', use_bias=False, name='res{0}{1}_branch2b'.format(stage, substage))(conv)
    conv=Conv2D(filters*4, 1, use_bias=False, name='res{0}{1}_branch2c'.format(stage, substage))(conv)    
    add=Add(name='res{0}{1}_'.format(stage, substage))([inp, conv])
    return ReLU(name='res{0}{1}'.format(stage, substage))(add)

def res_block_ext(inp, stage):    
    filters=inp.shape.as_list()[-1]//2
    strds=(2,2)
    if filters==32:
        filters=filters*2
        strds=(1,1)        
    conv=Conv2D(filters, 1, strides=strds, activation='relu', use_bias=False, name='res{0}a_branch2a'.format(stage))(inp)    
    conv=Conv2D(filters, 3, padding='same', activation='relu', use_bias=False, name='res{0}a_branch2b'.format(stage))(conv)     
    conv=Conv2D(filters*4, 1, use_bias=False, name='res{0}a_branch2c'.format(stage))(conv)    
    inp1=Conv2D(filters*4, 1, strides=strds, use_bias=False, name='res{0}a_branch1'.format(stage))(inp)    
    
    add=Add(name='res{0}a_'.format(stage))([conv,inp1])
    return ReLU(name='res{0}a'.format(stage))(add)

def get_res50(inp_shape=(500,500,3)):
    inp=Input(shape=inp_shape)
    conv=Conv2D(64, 7, padding='same', activation='relu', name='conv1')(inp)    
    out=MaxPool2D(strides=2, padding='same', name='pool1')(conv)    
    outputs=[conv]
    
    for num, blocks in enumerate([2,3,5,2]):
        stage=num+2
        out=res_block_ext(out, stage)
        for substage in 'bcdef'[:blocks]:
            out=res_block_simple(out, stage, substage)
        outputs.append(out)
    return Model(inputs=inp,outputs=outputs)


#Outlines and orientations branches
class Crop(layers.Layer):
    """cropping of target layer by dimensions of template layer according with stride of prev decov layer
    EXAMPLE:
    cropped_lr=Crop(name='cropped_layer')(target, templ)
    """
    def __init__(self, pad, **kwargs):
        super().__init__(**kwargs)
        #         super(Crop, self).__init__(**kwargs)
        self.pad=pad        
        #     def get_config(self):
        #         config = super().get_config().copy()
        #         config.update({
        #             'pad': self.pad,            
        #         })
        #         return config
    
    def compute_output_shape(self, input_shape):
        sh=(input_shape[1][0],input_shape[1][1],input_shape[1][2],input_shape[0][3])
        return sh
        
    def call(self, x, trainable=False):
        x_shape = tf.shape(x[0])
        templ_sh= tf.shape(x[1])  
        offsets = [0, self.pad-1, self.pad-1, 0]
        size = [-1, templ_sh[1], templ_sh[2], x_shape[3]]        
        
        x_crop = tf.slice(x[0], offsets, size)
        x_crop.set_shape(self.compute_output_shape([x[0].shape, x[1].shape]))
        return x_crop    
    
     
def DSN_deconv(lr, inp_lr):
    lr_name=lr.name.split('/')[0]
    lr_num=[int(s) for s in lr_name if s.isdigit()][0]    
    con=Conv2D(1,1, name='score-dsn{0}'.format(lr_num))(lr)
    st=1
    if lr_num>1:
        st=2**(lr_num-1)
        ks=st*2
        con=Conv2DTranspose(1,ks,(st,st), name='upsample_{0}'.format(st))(con)
    con=Crop(pad=st,name='crop{0}'.format(lr_num))([con, inp_lr])
    return con     

def get_outlines_net(res_inp, res_outp):
    DSN=[DSN_deconv(lr, res_inp) for lr in res_outp] 
    
    con1=Concatenate(name="concat1")(DSN[:-1])
    con1=Conv2D(1,1, name='new-score-weighting1')(con1)
    con1=Activation('sigmoid', name='sigmoid-fuse_scale_2.0')(con1)
    
    con2=Concatenate(name="concat3")(DSN[1:])
    con2=Conv2D(1,1, name='new-score-weighting3')(con2)
    con2=Activation('sigmoid', name='sigmoid-fuse_scale_0.5')(con2)
    return [con2, con1]
    #return {'sigmoid-fuse_scale_2.0':con1, 'sigmoid-fuse_scale_0.5':con2}
    
def orient_deconv(lr, inp_lr, orient):
    lr_name=lr.name.split('/')[0]
    lr_num=[int(s) for s in lr_name if s.isdigit()][0]    
    con=Conv2D(32,3, padding='same', name='{0}_or8_{1}'.format(lr_name, orient))(lr)
    con=Conv2D(4,3, padding='same', name='{0}_4_or8_{1}'.format(lr_name, orient))(con)
    
    st=1
    if lr_num>1:
        st=2**(lr_num-1)
        ks=st*2
#         con=Conv2DTranspose(4,ks,(st,st), name='{0}_4_or8_{1}-up'.format(lr_name, orient))(con)
        con=Conv2DTranspose(4,ks,(st,st), name='upsample_{0}_or8_{1}'.format(st, orient))(con)
                
    con=Crop(pad=st, name='{0}_4_or8_{1}_cropped'.format(lr_name, orient))([con, inp_lr])
    return con

def get_orientations_net(res_inp, res_outp):
    outputs=[]
    #outputs={}
    
    for i in range(1,9): 
        orients=[orient_deconv(lr, res_inp, i) for lr in res_outp]
        con=Concatenate(name='concat-upscore_or8_{0}'.format(i))(orients)
        #         con=Conv2D(1,3, padding='same', name='upscore-fuse_or8_{0}'.format(i))(con)
        
        con=Conv2D(1,3, padding='same', name='score-or8_{0}'.format(i))(con)
        
        con=Activation('sigmoid', name='sigmoid-fuse_or8_{0}'.format(i))(con)
        outputs.append(con)
        #outputs['sigmoid-fuse_or8_{0}'.format(i)]=con
    return outputs


def get_COB_model(weight_path='COB_PASCALContext_trainval.h5', input_shape=(None,None,3)):
    
    '''Make tf COB model with specified weights:
    
    weight_path - path to the weights h5 file ('COB_PASCALContext_trainval.h5' by default),
    input_shape - shape of input image, by default (None, None,3)    
    '''
    
    import os.path
    assert os.path.isfile(weight_path), '{0} - not correct weigths path'.format(weight_path)
    res50=get_res50(inp_shape=input_shape)
    res_inp=res50.inputs
    res_out=res50.output
    orientations=get_orientations_net(res_inp[0], res_out)
    outlines=get_outlines_net(res_inp[0], res_out)

    orient_model=Model(res_inp, orientations+outlines)
    orient_model.load_weights(weight_path)
    return orient_model
