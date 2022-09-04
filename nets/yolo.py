from keras.layers import (Concatenate, Input, Lambda, UpSampling2D, MaxPooling2D, Add, BatchNormalization, Conv2D,
                          ZeroPadding2D)
from keras.models import Model

from nets.backbone import (DarknetConv2D, DarknetConv2D_BN, DarknetConv2D_BN_SiLU, Transition_Block, Multi_Concat_Block, SiLU,
                             darknet_body)
from nets.yolo_training import yolo_loss


def SPPCSPC(x, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), name=""):
    c_ = int(2 * c2 * e)  # hidden channels
    x1 = DarknetConv2D_BN_SiLU(c_, (1, 1), name = name + '.cv1')(x)
    x1 = DarknetConv2D_BN_SiLU(c_, (3, 3), name = name + '.cv3')(x1)
    x1 = DarknetConv2D_BN_SiLU(c_, (1, 1), name = name + '.cv4')(x1)
    
    y1 = Concatenate(axis=-1)([x1] + [MaxPooling2D(pool_size=(m, m), strides=(1, 1), padding='same')(x1) for m in k])
    y1 = DarknetConv2D_BN_SiLU(c_, (1, 1), name = name + '.cv5')(y1)
    y1 = DarknetConv2D_BN_SiLU(c_, (3, 3), name = name + '.cv6')(y1)
    
    y2 = DarknetConv2D_BN_SiLU(c_, (1, 1), name = name + '.cv2')(x)
    out = Concatenate(axis=-1)([y1, y2])
    out = DarknetConv2D_BN_SiLU(c2, (1, 1), name = name + '.cv7')(out)
    
    return out

def RepConv(x, c2, name=""):
    x1 = Conv2D(c2, (3, 3), name = name + '.rbr_dense.0', use_bias=False, padding='same')(x)
    x1 = BatchNormalization(momentum = 0.97, epsilon = 0.001, name = name + '.rbr_dense.1')(x1)
    x2 = Conv2D(c2, (1, 1), name = name + '.rbr_1x1.0', use_bias=False, padding='same')(x)
    x2 = BatchNormalization(momentum = 0.97, epsilon = 0.001, name = name + '.rbr_1x1.1')(x2)
    
    out = Add()([x1, x2])
    out = SiLU()(out)
    return out

#---------------------------------------------------#
#   Panet网络的构建，并且获得预测结果
#---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes, phi):
    #-----------------------------------------------#
    #   定义了不同yolov7版本的参数
    #-----------------------------------------------#
    transition_channels = {'l' : 32, 'x' : 40}[phi]
    block_channels      = 32
    panet_channels      = {'l' : 32, 'x' : 64}[phi]
    e       = {'l' : 2, 'x' : 1}[phi]
    n       = {'l' : 4, 'x' : 6}[phi]
    ids     = {'l' : [-1, -2, -3, -4, -5, -6], 'x' : [-1, -3, -5, -7, -8]}[phi]

    inputs      = Input(input_shape)
    #---------------------------------------------------#   
    #   生成主干模型，获得三个有效特征层，他们的shape分别是：
    #   80, 80, 256
    #   40, 40, 1024
    #   20, 20, 1024
    #---------------------------------------------------#
    feat1, feat2, feat3 = darknet_body(inputs, transition_channels, block_channels, n, phi)

    # 20, 20, 1024 -> 20, 20, 512
    P5          = SPPCSPC(feat3, transition_channels * 16, name="sppcspc")
    P5_conv     = DarknetConv2D_BN_SiLU(transition_channels * 8, (1, 1), name="conv_for_P5")(P5)
    P5_upsample = UpSampling2D()(P5_conv)
    P4          = Concatenate(axis=-1)([DarknetConv2D_BN_SiLU(transition_channels * 8, (1, 1), name="conv_for_feat2")(feat2), P5_upsample])
    P4          = Multi_Concat_Block(P4, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids, name="conv3_for_upsample1")

    P4_conv     = DarknetConv2D_BN_SiLU(transition_channels * 4, (1, 1), name="conv_for_P4")(P4)
    P4_upsample = UpSampling2D()(P4_conv)
    P3          = Concatenate(axis=-1)([DarknetConv2D_BN_SiLU(transition_channels * 4, (1, 1), name="conv_for_feat1")(feat1), P4_upsample])
    P3          = Multi_Concat_Block(P3, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids, name="conv3_for_upsample2")
        
    P3_downsample = Transition_Block(P3, transition_channels * 4, name="down_sample1")
    P4 = Concatenate(axis=-1)([P3_downsample, P4])
    P4 = Multi_Concat_Block(P4, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids, name="conv3_for_downsample1")

    P4_downsample = Transition_Block(P4, transition_channels * 8, name="down_sample2")
    P5 = Concatenate(axis=-1)([P4_downsample, P5])
    P5 = Multi_Concat_Block(P5, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids, name="conv3_for_downsample2")
    
    if phi == "l":
        P3 = RepConv(P3, transition_channels * 8, name="rep_conv_1")
        P4 = RepConv(P4, transition_channels * 16, name="rep_conv_2")
        P5 = RepConv(P5, transition_channels * 32, name="rep_conv_3")
    else:
        P3 = DarknetConv2D_BN_SiLU(transition_channels * 8, (3, 3), strides=(1, 1), name="rep_conv_1")(P3)
        P4 = DarknetConv2D_BN_SiLU(transition_channels * 16, (3, 3), strides=(1, 1), name="rep_conv_2")(P4)
        P5 = DarknetConv2D_BN_SiLU(transition_channels * 32, (3, 3), strides=(1, 1), name="rep_conv_3")(P5)

    # len(anchors_mask[2]) = 3
    # 5 + num_classes -> 4 + 1 + num_classes
    # 4是先验框的回归系数，1是sigmoid将值固定到0-1，num_classes用于判断先验框是什么类别的物体
    # bs, 20, 20, 3 * (4 + 1 + num_classes)
    out2 = DarknetConv2D(len(anchors_mask[2]) * (5 + num_classes), (1, 1), strides = (1, 1), name = 'yolo_head_P3')(P3)
    out1 = DarknetConv2D(len(anchors_mask[1]) * (5 + num_classes), (1, 1), strides = (1, 1), name = 'yolo_head_P4')(P4)
    out0 = DarknetConv2D(len(anchors_mask[0]) * (5 + num_classes), (1, 1), strides = (1, 1), name = 'yolo_head_P5')(P5)
    return Model(inputs, [out0, out1, out2])

def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    model_loss  = Lambda(
        yolo_loss, 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
        arguments       = {
            'input_shape'       : input_shape, 
            'anchors'           : anchors, 
            'anchors_mask'      : anchors_mask, 
            'num_classes'       : num_classes, 
            'label_smoothing'   : label_smoothing, 
            'balance'           : [0.4, 1.0, 4],
            'box_ratio'         : 0.05,
            'obj_ratio'         : 1 * (input_shape[0] * input_shape[1]) / (640 ** 2), 
            'cls_ratio'         : 0.5 * (num_classes / 80)
        }
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model
