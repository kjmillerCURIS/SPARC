import os
import sys
import argparse
import copy
import numpy as np
import pickle
import re
from sklearn.decomposition import PCA
from tqdm import tqdm


#minor classname remappings for COCO dataset
COCO_COMPOUND_CLASSNAMES_RENAMER = {'food bowl' : 'bowl'}
COCO_SINGLETON_CLASSNAMES_RENAMER = {'motor bike' : 'motorcycle', 'aeroplane' : 'airplane'}


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def depluralize(s):
    if s[-1] == 's':
        return s[:-1]
    else:
        return s


def stringmatch_to_compoundprompt(classnames, compoundprompt, is_voc=False):
    for classname in classnames:
        assert(re.sub(r'[^a-zA-Z]+', ' ', classname.lower()).strip() == classname)
        assert('  ' not in classname)

    sorted_classnames = sorted(classnames, key = lambda c: len(c), reverse=True)
    matches = []
    substrate = compoundprompt.lower()
    if is_voc:
        substrate = substrate.replace('potted plant', 'pottedplant').replace('dining table', 'diningtable').replace('tv monitor', 'tvmonitor')

    substrate = ' ' + re.sub(r'[^a-zA-Z]+', ' ', substrate) + ' '
    for classname in sorted_classnames:
        flag = False
        for c in sorted(set([classname, depluralize(classname), classname + 's', classname + 'es', classname.replace('person', 'people').replace('child', 'children').replace('man', 'men').replace('foot', 'feet').replace('goose', 'geese').replace('mouse', 'mice').replace('die', 'dice').replace('tooth', 'teeth').replace('louse', 'lice').replace('leaf', 'leaves').replace('wolf', 'wolves').replace('knife', 'knives').replace('cactus', 'cacti').replace('shelf', 'shelves').replace('calf', 'calves')])):
            if ' ' + c + ' ' in substrate:
                flag = True
                substrate = substrate.replace(' ' + c + ' ', ' ! ')

        if flag:
            matches.append(classname)

    return matches


def check_classname2compoundprompts(classname2compoundprompts, classnames):
    for classname in sorted(classnames):
        if len(classname2compoundprompts[classname]) < 4:
            print('Low number of compound prompts for class "%s": %s'%(classname, ', '.join(['"%s"'%(p) for p in classname2compoundprompts[classname]])))


def load_data(input_dir, dataset_name, model_type):
    assert(dataset_name in ['VOC2007', 'COCO2014', 'NUSWIDE'])
    is_voc = (dataset_name == 'VOC2007')
    is_coco = (dataset_name == 'COCO2014')

    #load pkl files
    ensemble_single_filename = os.path.join(input_dir, '%s_test_baseline_zsclip_cossims_%s_ensemble_80.pkl'%(dataset_name, model_type))
    with open(ensemble_single_filename, 'rb') as f:
        d_ensemble_single = pickle.load(f)

    simple_single_and_compound_filename = os.path.join(input_dir, '%s_test_simple_single_and_compound_cossims_%s.pkl'%(dataset_name, model_type))
    with open(simple_single_and_compound_filename, 'rb') as f:
        d_simple_single_and_compound = pickle.load(f)

    if is_coco:
        #class renaming
        d_ensemble_single['classnames'] = [(COCO_SINGLETON_CLASSNAMES_RENAMER[c] if c in COCO_SINGLETON_CLASSNAMES_RENAMER else c) for c in d_ensemble_single['classnames']]
        d_simple_single_and_compound['simple_single_and_compound_prompts'][:len(d_ensemble_single['classnames'])] = [(COCO_COMPOUND_CLASSNAMES_RENAMER[c] if c in COCO_COMPOUND_CLASSNAMES_RENAMER else c) for c in d_simple_single_and_compound['simple_single_and_compound_prompts'][:len(d_ensemble_single['classnames'])]]

    #get classnames, gts, compoundprompts, double-check everything
    classnames = d_ensemble_single['classnames']
    num_classes = len(classnames)
    assert(d_simple_single_and_compound['simple_single_and_compound_prompts'][:num_classes] == classnames)

    gts_arr = d_ensemble_single['gts']
    assert(np.all(d_simple_single_and_compound['gts'] == gts_arr))
    compoundprompts = d_simple_single_and_compound['simple_single_and_compound_prompts'][num_classes:]
    assert(all([' ' in p for p in compoundprompts]))

    #do string-matching stuff
    classname2compoundprompts = {c : [] for c in classnames}
    compoundprompt2classnames = {}
    for compoundprompt in tqdm(compoundprompts):
        matches = stringmatch_to_compoundprompt(classnames, compoundprompt, is_voc=is_voc)
        compoundprompt2classnames[compoundprompt] = matches
        for classname in matches:
            classname2compoundprompts[classname].append(compoundprompt)

    check_classname2compoundprompts(classname2compoundprompts, classnames)

    #get cossims, make structures
    ensemble_single_cossims_arr = d_ensemble_single['cossims']
    simple_single_cossims_arr = d_simple_single_and_compound['cossims'][:,:num_classes]
    compound_cossims_arr = d_simple_single_and_compound['cossims'][:,num_classes:]
    gts = [{c : v for c, v in zip(classnames, gts_row)} for gts_row in gts_arr]
    ensemble_single_cossims = [{c : v for c, v in zip(classnames, cossims_row)} for cossims_row in ensemble_single_cossims_arr]
    simple_single_cossims = [{c : v for c, v in zip(classnames, cossims_row)} for cossims_row in simple_single_cossims_arr]
    compound_cossims = [{p : v for p, v in zip(compoundprompts, cossims_row)} for cossims_row in compound_cossims_arr]

    lens = [len(classname2compoundprompts[c]) for c in sorted(classname2compoundprompts.keys())]
    lensnz = [l for l in lens if l > 0]

    #return everything
    return gts, ensemble_single_cossims, simple_single_cossims, compound_cossims, classname2compoundprompts, compoundprompt2classnames, d_ensemble_single['impaths']


def calibrate_by_row(ensemble_single_cossims, simple_single_cossims, compound_cossims, skip_rowcalibbase=False):
    ensemble_single_cossims_calib = []
    compound_cossims_calib = []
    for ensemble_single_row, simple_single_row, compound_row in zip(ensemble_single_cossims, simple_single_cossims, compound_cossims):
        ensemble_mean = np.mean([ensemble_single_row[k] for k in sorted(ensemble_single_row.keys())])
        ensemble_sd = np.std([ensemble_single_row[k] for k in sorted(ensemble_single_row.keys())], ddof=1)
        simple_mean = np.mean([simple_single_row[k] for k in sorted(simple_single_row.keys())])
        simple_sd = np.std([simple_single_row[k] for k in sorted(simple_single_row.keys())], ddof=1)
        if skip_rowcalibbase:
            ensemble_single_row_calib = ensemble_single_row
        else:
            ensemble_single_row_calib = {k : (ensemble_single_row[k]-ensemble_mean)/ensemble_sd for k in sorted(ensemble_single_row.keys())}
        compound_row_calib = {k : (compound_row[k] - simple_mean) / simple_sd for k in sorted(compound_row.keys())}
        ensemble_single_cossims_calib.append(ensemble_single_row_calib)
        compound_cossims_calib.append(compound_row_calib)

    return ensemble_single_cossims_calib, compound_cossims_calib


def calibrate_by_column(cossims):
    my_keys = sorted(cossims[0].keys())
    cossims_arr = np.array([[cossims_row[k] for k in my_keys] for cossims_row in cossims])
    means = np.mean(cossims_arr, axis=0, keepdims=True)
    sds = np.std(cossims_arr, axis=0, keepdims=True, ddof=1)
    cossims_arr = (cossims_arr - means) / sds
    cossims = [{k : cossims_arr_row[j] for j, k in enumerate(my_keys)} for cossims_arr_row in cossims_arr]
    return cossims


#return ensemble_single_cossims, compound_cossims
#yes, these are calibrated even though I call them cossims! That's the nomenclature now! Deal with it!
#there's no "calibration type", either you call this function or you don't
def calibrate_data(ensemble_single_cossims, simple_single_cossims, compound_cossims, skip_rowcalibbase=False):
    ensemble_single_cossims, compound_cossims = calibrate_by_row(ensemble_single_cossims, simple_single_cossims, compound_cossims, skip_rowcalibbase=skip_rowcalibbase)
    ensemble_single_cossims = calibrate_by_column(ensemble_single_cossims)
    compound_cossims = calibrate_by_column(compound_cossims)
    return ensemble_single_cossims, compound_cossims


#return ensemble_single_scores and allpcawsing_scores
#yes, this does the whole dataset, but just for one class
#no, this does NOT do calibration, be sure to do that beforehand (unless ablating it)
def predict_oneclass(classname, ensemble_single_cossims, compound_cossims, classname2compoundprompts):
    assert(len(ensemble_single_cossims) == len(compound_cossims))
    sorted_compound_scores_list = []
    ensemble_single_scores = []
    for ensemble_single_row, compound_row in zip(ensemble_single_cossims, compound_cossims):
        ensemble_single_score = ensemble_single_row[classname]
        compound_scores = np.array([compound_row[prompt] for prompt in classname2compoundprompts[classname]])
        sorted_compound_scores = np.sort(compound_scores) #use index -topk
        sorted_compound_scores_list.append(sorted_compound_scores)
        ensemble_single_scores.append(ensemble_single_score)

    if len(classname2compoundprompts[classname]) == 0:
        allpcawsing_scores = np.array(ensemble_single_scores)
    else:
        stuff_for_PCAwsing = np.array(sorted_compound_scores_list)
        stuff_for_PCAwsing = [stuff_for_PCAwsing[:,i] for i in range(stuff_for_PCAwsing.shape[1])]
        stuff_for_PCAwsing.append(np.array(ensemble_single_scores))
        allpcawsing_scores, _, __ = do_PCA_oneclass(*stuff_for_PCAwsing, use_avg_for_sign=True)

    return np.array(ensemble_single_scores), np.array(allpcawsing_scores)


def do_PCA_oneclass(*scores_list, use_avg_for_sign=False, normalize_direction_by_L1=True):
    assert(normalize_direction_by_L1)
    scores_arr = np.array(list(scores_list)).T
    assert(scores_arr.shape[0] > scores_arr.shape[1])
    my_pca = PCA()
    my_pca.fit(scores_arr)
    direction = my_pca.components_[0,:]
    if not (np.all(direction > 0) or np.all(direction < 0)):
        print('PCA direction with some negative weights: ' + str(direction))

    if use_avg_for_sign:
        if np.mean(direction) < 0.0:
            direction = -1 * direction
    else:
        if direction[0] < 0.0:
            direction = -1 * direction

    if normalize_direction_by_L1:
        direction = direction / np.sum(np.fabs(direction))

    pca_scores = np.squeeze(scores_arr @ direction[:,np.newaxis]) #no need to center, it's just a constant offset
    pca_direction = direction
    pca_explvars = my_pca.explained_variance_
    return pca_scores, pca_direction, pca_explvars


def append_to_results(method, mAP, APs, results, f):
    results[method] = {'mAP' : mAP, 'APs' : APs}
    f.write('%s,%f\n'%(method, mAP))
    print('%s: mAP=%f'%(method, mAP))


def run_SPARC(input_dir, dataset_name, model_type, output_prefix):
    print('load data...')
    gts, ensemble_single_cossims, simple_single_cossims, compound_cossims, classname2compoundprompts, compoundprompt2classnames, impaths = load_data(input_dir, dataset_name, model_type)
    classnames = sorted(gts[0].keys())

    #uncalibrated APs
    ensemble_single_scores_uncalib = {c : np.array([row[c] for row in ensemble_single_cossims]) for c in classnames}
    print('uncalibrated_APs...')
    ensemble_single_uncalibrated_APs = {}
    for classname in classnames:
        output = np.array([row[classname] for row in ensemble_single_cossims])
        target = np.array([row[classname] for row in gts])
        ensemble_single_uncalibrated_APs[classname] = 100.0 * average_precision(output, target)

    ensemble_single_uncalibrated_mAP = np.mean([ensemble_single_uncalibrated_APs[classname] for classname in classnames])

    print('normalize...')
    ensemble_single_cossims, compound_cossims = calibrate_data(ensemble_single_cossims, simple_single_cossims, compound_cossims)

    #get the scores, each as dict mapping from classname to 1D array
    #also get the uniform averaging scores
    print('rank and fuse...')
    ensemble_single_scores = {}
    allpcawsing_scores = {}
    allpcawsing_avg_scores = {}
    for classname in tqdm(classnames):
        ensemble_single_scores[classname], allpcawsing_scores[classname] = predict_oneclass(classname, ensemble_single_cossims, compound_cossims, classname2compoundprompts)
        allpcawsing_avg_scores[classname] = 0.5 * allpcawsing_scores[classname] + 0.5 * ensemble_single_scores[classname]

    #compute APs
    print('APs...')
    ensemble_single_APs, allpcawsing_avg_APs = {}, {}
    for classname in tqdm(classnames):
        gts_arr = np.array([gts_row[classname] for gts_row in gts])
        ensemble_single_APs[classname] = 100.0 * average_precision(ensemble_single_scores[classname], gts_arr)
        allpcawsing_avg_APs[classname] = 100.0 * average_precision(allpcawsing_avg_scores[classname], gts_arr)

    #compute mAP
    ensemble_single_mAP = np.mean([ensemble_single_APs[classname] for classname in classnames])
    allpcawsing_avg_mAP = np.mean([allpcawsing_avg_APs[classname] for classname in classnames])

    #save results
    print('saving results...')
    results_pkl_filename, results_csv_filename = output_prefix + '.pkl', output_prefix + '.csv'
    f = open(results_csv_filename, 'w')
    f.write('method,mAP\n')
    results = {}
    append_to_results('CLIP (baseline)', ensemble_single_uncalibrated_mAP, ensemble_single_uncalibrated_APs, results, f)
    append_to_results('Normalize only', ensemble_single_mAP, ensemble_single_APs, results, f)
    append_to_results('SPARC', allpcawsing_avg_mAP, allpcawsing_avg_APs, results, f)
    f.close()
    with open(results_pkl_filename, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SPARC pipeline given pre-computed CLIP cosine similarities.")
    
    parser.add_argument("--input_dir", type=str, help="Path to directory with pre-computed CLIP cosine similarities")
    parser.add_argument("--dataset_name", type=str, help="Dataset name (e.g. COCO2014, VOC2007, NUSWIDE)")
    parser.add_argument("--model_type", type=str, help="CLIP model architecture type (ViT-L14336px, ViT-L14, ViT-B32, RN50x64, RN50x16, RN50x4, RN101, RN50)")
    parser.add_argument("--output_prefix", type=str, help="Prefix for output file (.csv has mAPs, .pkl also has individual class APs)")
    
    args = parser.parse_args()
    run_SPARC(args.input_dir, args.dataset_name, args.model_type, args.output_prefix)
