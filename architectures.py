import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
import torch.nn.functional as F

CONV_BLOCK='convblock'
DECONV_BLOCK='deconvblock'
ADP_AVG_POOL='adaptive_avg_pool'
FLT='flt'
DENSE='dense'
MAX_POOL='max_pool'

LAYERS_TYPE_LIST=[CONV_BLOCK,DECONV_BLOCK,ADP_AVG_POOL,FLT,DENSE,MAX_POOL]
ARCHITECTURES={
    'shared':[
                [
                {'type':CONV_BLOCK,'in_ch': 1, 'out_ch': 8,'kernel_size': 5,'stride': 1,'drop_rate': 0.1},
                {'type':CONV_BLOCK,'in_ch': 8, 'out_ch': 16,'kernel_size': 1,'stride': 1,'drop_rate': 0.1},
                {'type':CONV_BLOCK,'in_ch': 16, 'out_ch': 32,'kernel_size': 5,'stride': 1,'drop_rate': 0.1},
                {'type':CONV_BLOCK,'in_ch': 32, 'out_ch': 32,'kernel_size': 1,'stride': 1,'drop_rate': 0.1},
                ],
                [
                {'type':CONV_BLOCK,'in_ch': 1, 'out_ch': 8,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
                {'type':CONV_BLOCK,'in_ch': 8, 'out_ch': 16,'kernel_size': 1,'stride': 1,'drop_rate': 0.1},
                {'type':CONV_BLOCK,'in_ch': 16, 'out_ch': 32,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
                {'type':CONV_BLOCK,'in_ch': 32, 'out_ch': 32,'kernel_size': 1,'stride': 1,'drop_rate': 0.1}, 
                ]
            ],
    7:{
    2:{
        'coarse': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},      
        ],

        'fine_0': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},  
        ],
        'fine_1': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT}, 
        ],
    },
    3:{
        'coarse': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},      
        ],

        'fine_0': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},  
        ],
        'fine_1': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT}, 
        ],
        'fine_2': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT}, 
        ],
    },
    4:{
        'coarse': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},      
        ],

        'fine_0': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},  
        ],
        'fine_1': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT}, 
        ],
        'fine_2': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT}, 
        ],
    },
    5:{
        'coarse': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},      
        ],

        'fine_0': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},  
        ],
        'fine_1': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT}, 
        ],
    },
    6:{
        'coarse': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},      
        ],

        'fine_0': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},  
        ],

    },
    },
    8:{
    2:{
        'coarse': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},      
        ],

        'fine_0': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},  
        ],
        'fine_1': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT}, 
        ],
    },
    3:{
        'coarse': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},      
        ],

        'fine_0': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},  
        ],
        'fine_1': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT}, 
        ],
        'fine_2': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT}, 
        ],
    },
    4:{
        'coarse': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},      
        ],

        'fine_0': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},  
        ],
        'fine_1': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT}, 
        ],
        'fine_2': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT}, 
        ],
    },
    5:{
        'coarse': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},


            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},      
        ],

        'fine_0': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},  
        ],
        'fine_1': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT}, 
        ],
    },
    6:{
        'coarse': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},      
        ],

        'fine_0': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},  
        ],
        'fine_1': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},  
        ],

    },
    7:{
        'coarse': [
            {'type':MAX_POOL,'kernel_size': 3,'stride': 3},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},      
        ],

        'fine_0': [
            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':DECONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':MAX_POOL,'kernel_size': 2,'stride': 2},

            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},
            {'type':CONV_BLOCK,'in_ch': 64, 'out_ch': 64,'kernel_size': 3,'stride': 1,'drop_rate': 0.1},

            {'type':ADP_AVG_POOL,'out_shape': (1,1)},
            {'type':FLT},  
        ],

    },
    }
}
N_EMOTION_CLASSES___COARSE_FINE_EMOTION_MAPPER_DICT={
    7:{
        2:{
            0:{
                0:0,
                1:1,
                2:2,
                3:3,
                4:5

                },

            1:{
                0:4,
                1:6
            }
        },
        3:{
            0:{
                0:3,
                1:4,
                2:6
            },

            1:{
                0:0,
                1:1,
            },

            2:{
                0:2,
                1:5
            }
        },
        4:{
            0:{
                0:0,
                1:1
            },

            1:{
                0:2,
                1:5
            },

            2:{
                0:3,
                1:6
            },
            3:{
                0:4
            }
        },
        5:{
            0:{
                0:0,
                1:1
            },

            1:{
                0:2,
                1:5
            },

            2:{
                0:3
            },
            3:{
                0:4
            },
            4:{
                0:6
            }
        },
        6:{
            0:{
                0:2,
                1:5
            },

            1:{
                0:0
            },

            2:{
                0:1
            },
            3:{
                0:3
            },
            4:{
                0:4
            },
            5:{
                0:6
            }
        },
    },
    8:{
        2:{
            0:{
                0:0,
                1:1,
                2:4,
                3:7,
                },

            1:{
                0:2,
                1:3,
                2:5,
                3:6
            }
        },
        3:{
            0:{
                0:0,
                1:1,
                2:4,
                3:7,
            },

            1:{
                0:2,
                1:5,
            },

            2:{
                0:3,
                1:6
            }
        },
        4:{
            0:{
                0:0,
                1:1,
                2:7
            },

            1:{
                0:2,
                1:5
            },

            2:{
                0:4,
                1:6
            },
            3:{
                0:3
            }
        },
        5:{
            0:{
                0:0,
                1:1,
                2:7
            },

            1:{
                0:2,
                1:5
            },

            2:{
                0:3
            },
            3:{
                0:4
            },
            4:{
                0:6
            }
        },
        6:{
            0:{
                0:0,
                1:7
            },

            1:{
                0:2,
                1:5
            },

            2:{
                0:1
            },
            3:{
                0:3
            },
            4:{
                0:4
            },
            5:{
                0:6
            }
        },
        7:{
            0:{
                0:0,
                1:7
            },

            1:{
                0:1,
            },

            2:{
                0:2
            },
            3:{
                0:3
            },
            4:{
                0:4
            },
            5:{
                0:5
            },
            6:{
                0:6
            }
        },
    }

}

def get_coarse_fine_emotion_mapper(n_emotion_classes,n_coarse_classes):
    if n_emotion_classes == 7:
        return N_EMOTION_CLASSES___COARSE_FINE_EMOTION_MAPPER_DICT[n_emotion_classes][n_coarse_classes]
    else:
        return N_EMOTION_CLASSES___COARSE_FINE_EMOTION_MAPPER_DICT[n_emotion_classes][n_coarse_classes]

def get_block(layer_info_dict):
    layer_type=layer_info_dict['type']

    match layer_type:

        case lt if lt == LAYERS_TYPE_LIST[0]:
            return nn.Sequential(*[
                nn.Conv2d(in_channels=layer_info_dict['in_ch'],out_channels=layer_info_dict['out_ch'],kernel_size=layer_info_dict['kernel_size'],stride=layer_info_dict['stride'],bias=False),
                nn.BatchNorm2d(layer_info_dict['out_ch']),
                nn.ReLU(),
                nn.Dropout2d(layer_info_dict['drop_rate']),
            ])
        
        case lt if lt == LAYERS_TYPE_LIST[1]:
            return nn.Sequential(*[
                nn.ConvTranspose2d(in_channels=layer_info_dict['in_ch'],out_channels=layer_info_dict['out_ch'],kernel_size=layer_info_dict['kernel_size'],stride=layer_info_dict['stride'],bias=False),
                nn.BatchNorm2d(layer_info_dict['out_ch']),
                nn.ReLU(),
                nn.Dropout2d(layer_info_dict['drop_rate']),
            ])
        
        case lt if lt == LAYERS_TYPE_LIST[2]:
            return nn.AdaptiveAvgPool2d(layer_info_dict['out_shape'])
        case lt if lt == LAYERS_TYPE_LIST[3]:
            return nn.Flatten()
        case lt if lt == LAYERS_TYPE_LIST[4]:
            return nn.Linear(layer_info_dict['in_ch'],layer_info_dict['out_ch'] )
        case lt if lt == LAYERS_TYPE_LIST[5]:
            return nn.MaxPool2d(layer_info_dict['kernel_size'],layer_info_dict['stride'] )

def get_sub_module(layers_list):
    result=[]
    for layer in layers_list:
        result.append(get_block(layer))
    return result

def get_architecture(n_cf_classes_list,n_emotion_classes):
    n_coarse=n_cf_classes_list[0]
    arch={}
    arch['shared']=ARCHITECTURES['shared']
    arch.update(ARCHITECTURES[n_emotion_classes][n_coarse])

    for idx, n_class in enumerate(n_cf_classes_list):
        if idx == 0:
            arch['coarse'].append({'type':DENSE,'in_ch': 64,'out_ch':n_class})
        else:
            arch['fine_'+str(idx-1)].append({'type':DENSE,'in_ch': 64,'out_ch':n_class})

    torch_arch={}
    for sub_module in arch:
        if sub_module == 'shared':
            torch_arch[sub_module]=[]
            for shared_branch in arch[sub_module]:
                torch_arch[sub_module].append(get_sub_module(shared_branch))
        else:
            torch_arch[sub_module]=get_sub_module(arch[sub_module])
                
    return torch_arch


