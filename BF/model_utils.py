import torch
import numpy as np

from batchflow import Dataset, FilesIndex, Pipeline, B, V, D
from batchflow.models.torch import TorchModel, UNet

from preprocessing import (find_optimal_thr, list_crop, assemble_imgs,
                           filter_prediction, calculate_distance)

def create_config(size=40, weight=30., filters=8, num_blocks=4, kernel_size=3):
    inputs_config = {
    'images': {'shape': (1, size, size)}, 
    'masks': {'shape': (size, size),
              'classes': 2,
              'data_format': 'f',
              'name': 'targets'}
    }

    w = torch.Tensor([1., weight]).to('cuda')
    config = {
        'loss': {'name':'ce', 'weight': w},
        'inputs': inputs_config,
        'initial_block/inputs': 'images',
        'optimizer': ('Adam', {'lr': 0.001}),
        'head/num_classes': 2, 
        'body/num_blocks': num_blocks,
        'body/filters': [filters*2**i for i in range(num_blocks)], 
        'body/encoder': dict(layout='cna cna', kernel_size=kernel_size),
        'body/decoder': dict(layout='cna cna', kernel_size=kernel_size),
        'decay': ('exp', {'gamma': 0.99}),
        'n_iters': 58,
        'device': 'gpu:0',
    }
    return config

def create_tr_te_ppl(dset, config, model, prob=.5, size=40):
    prep_pipeline = (Pipeline()
                .load('fermi', components=['images', 'points'])
                .generate_masks(src='points', dst='masks')
                .normalize(src='images') 
                .random_crop_near_points(prob=prob, output_size=size, src=['images', 'masks'])
                .prepare_tensors('images', add_dim=True)
                .prepare_tensors('masks')
    )
    train_ppl = (prep_pipeline + (Pipeline()
                                  .init_model('dynamic', model, 'model', config)
                                  .init_variable('loss', default=[])
                                  .train_model('model', B('images'), B('masks'), 
                                               fetches='loss', save_to=V('loss',
                                                                         mode='a')))
                ) << dset[0]

    test_ppl = (prep_pipeline + (Pipeline()
                                 .import_model('model', train_ppl)
                                 .init_variable('loss', default=[])
                                 .init_variable('targets', default=[])
                                 .init_variable('predictions', default=[])
                                 .init_variable('images', default=[])
                                 .predict_model('model', B('images'), targets=B('masks'), 
                                                fetches=['predictions', 'loss'], 
                                                save_to=[V('predictions', mode='a'),
                                                         V('loss', mode='a')])
                                 .update(V('images', mode='a'), B('images'))
                                 .update(V('targets', mode='a'), B('masks')))
                ) << dset[1]
    return train_ppl, test_ppl

def train_models(train_ppl, test_ppl, n_epochs=50, save_data=False, save=False):
    train_loss = []
    test_loss = []
    targets = []
    preds = []
    for i in range(n_epochs):
        train_ppl.run(64, n_epochs=1, shuffle=True, drop_last=True)
        test_ppl.run(64, n_epochs=1, shuffle=True, drop_last=True)
        train_loss.append(np.mean(train_ppl.v('loss')))
        test_loss.append(np.mean(test_ppl.v('loss')))
        if save_data:
            targets.append(test_ppl.v('targets'))
            preds.append(test_ppl.v('predictions'))
    if save:
        print('Model have been saved to: {}'.format(save))
        train_ppl.save_model('model', save)
    return train_loss, test_loss, targets, preds

def load_data(dataset):
    load_ppl = (Pipeline()
                .load('fermi', components=['images', 'points'])
                .generate_masks(src='points', dst='masks')
                .normalize(src='images') 
                .prepare_tensors('images', add_dim=True)
                .prepare_tensors('masks')
                .init_variable('imag', default=[])
                .init_variable('mask', default=[])
                .init_variable('ix', default=[])
                .update(V('ix', mode='a'), B('ix'))
                .update(V('mask', mode='a'), B('masks'))
                .update(V('imag', mode='a'), B('images'))
                )
    test_data_ppl = (load_ppl << dataset).run(1, n_epochs=1,
                                              shuffle=False,
                                              drop_last=False)
    images = np.concatenate(np.concatenate(test_data_ppl.v('imag')))
    masks = np.concatenate(test_data_ppl.v('mask'))
    return images, masks

def find_optimal_params(model, dataset, size, step):
    threshold_list = np.linspace(.2, .9, 20)
    images, masks = load_data(dataset)
    g_dist, threshold = find_optimal_thr(model, images, masks, threshold_list, size, step)
    dist = calc_dist(model, images, masks, threshold, size, step)
    return dist, g_dist, threshold


def calc_dist(model, images, masks, tr, size, step):
    dist = []
    preds = []
    ans = []
    points = []
    for img, msk in list(zip(images, masks)):
        img_tens = torch.Tensor(list_crop(img.reshape(1, 200, 200), size, step))
        img_sigm = (torch.sigmoid(model.model(img_tens.to('cuda'))).cpu()
                    .detach().numpy().transpose(0, 2, 3, 1)[:,:,:,1])
        assemble_img = assemble_imgs(img_sigm, (200, 200), step)
        pr = np.array(assemble_img > tr)
        filt = filter_prediction(pr)
        preds.append(filt)
        img = np.array(np.where(filt > tr)).T
        mask = np.array(np.where(msk)).T
        ans.append(img)
        points.append(mask)
        dist.append(calculate_distance(img, mask))
    return dist

def calc_table(preds, targ, size=(200, 200)):
    false_pos = []
    false_neg = []
    true_pos = []
    true_neg = []
    all_points = size[0] * size[1]

    for pred, point in zip(preds, targ):
        fn_points = np.arange(len(point)) + 1
        fp = 0
        tp = 0
        fn = 0
        tn = 0
        if len(pred) == 0:
            if len(point) == 0:
                tn = all_points
            else:
                fn += len(point)
                tn += all_points - fn
        elif len(point) == 0:
            if len(pred) == 0:
                tn = all_points
            else:
                fp += len(pred)
                tn += all_points - fp
        else:
            for p in pred:
                closes = np.argmin(np.sum(np.abs(p - point), axis=1))
                if np.sum(np.abs(p - point[closes])) < 5:
                    tp += 1
                    fn_points[closes] = 0
                else:
                    fp += 1
        fn += len(np.where(fn_points != 0)[0])
        tn = all_points - tp - fp - fn
        false_pos.append(fp)
        false_neg.append(fn)
        true_pos.append(tp)
        true_neg.append(tn)
    return np.sum(false_pos), np.sum(false_neg), np.sum(true_pos), np.sum(true_neg)

def f1_score(tp, fp, fn):
    return (2*tp)/(2*tp+fp+fn)

def TPR(tp, fn):
    return tp/(tp+fn)

def FPR(tn, fp):
    return fp/(fp+tn)
