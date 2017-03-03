from caffe import layers as L
from caffe import params as P

n = caffe.NetSpec()


# helper functions for common structures
def conv_relu(bottom, ks, nout, weight_init='gaussian', weight_std=0.01, bias_value=0, mult=1, stride=1, pad=0,
              group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         weight_filler=dict(type=weight_init, mean=0.0, std=weight_std),
                         bias_filler=dict(type='constant', value=bias_value),
                         param=[dict(lr_mult=mult, decay_mult=mult), dict(lr_mult=2 * mult, decay_mult=0 * mult)])
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def FCN(images_lmdb, labels_lmdb, batch_size, include_acc=False):
    # net definition
    n.data = L.Data(source=images_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=1,
                    transform_param=dict(crop_size=0, mean_value=[77], mirror=False))
    n.label = L.Data(source=labels_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=1)
    n.conv1, n.relu1 = conv_relu(n.data, ks=5, nout=100, stride=2, pad=50, bias_value=0.1)
    n.pool1 = max_pool(n.relu1, ks=2, stride=2)
    n.conv2, n.relu2 = conv_relu(n.pool1, ks=5, nout=200, stride=2, bias_value=0.1)
    n.pool2 = max_pool(n.relu2, ks=2, stride=2)
    n.conv3, n.relu3 = conv_relu(n.pool2, ks=3, nout=300, stride=1, bias_value=0.1)
    n.conv4, n.relu4 = conv_relu(n.relu3, ks=3, nout=300, stride=1, bias_value=0.1)
    n.drop = L.Dropout(n.relu4, dropout_ratio=0.1, in_place=True)
    n.score_classes, _ = conv_relu(n.drop, ks=1, nout=2, weight_std=0.01, bias_value=0.1)
    n.upscore = L.Deconvolution(n.score_classes)
    n.score = L.Crop(n.upscore, n.data)
    n.loss = L.SoftmaxWithLoss(n.score, n.label, loss_param=dict(normalize=True))

    if include_acc:
        n.accuracy = L.Accuracy(n.score, n.label)
        return n.to_proto()
    else:
        return n.to_proto()


def make_nets():
    header = 'name: "FCN"\nforce_backward: true\n'
    with open('fcn_train.prototxt', 'w') as f:
        f.write(header + str(FCN('train_images_lmdb/', 'train_labels_lmdb/', batch_size=1, include_acc=False)))
    with open('fcn_test.prototxt', 'w') as f:
        f.write(header + str(FCN('val_images_lmdb/', 'val_labels_lmdb/', batch_size=1, include_acc=True)))


if __name__ == '__main__':
    make_nets()
