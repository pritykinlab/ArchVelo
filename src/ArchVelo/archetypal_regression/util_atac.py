def get_types(peaks, peaks_to_genes):
    ann_peaks = []
    for peak in peaks: 
        cur_ann = peaks_to_genes.loc[peak]
        peak_type = cur_ann['type']
        ann_peaks.append(peak_type)
    return ann_peaks

def collapse_types_func(arr):
    arr = arr.T
    arr['type'] = ['_'.join(x.split('_')[:3]) for x in arr.index]
    arr['Motif'] = [x.split('_')[3] if len(x.split('_'))>3 else '' for x in arr.index] 
    arr.set_index(['type', 'Motif'], append = True, inplace = True)
    arr = arr.groupby('Motif').sum()
    arr = arr.T
    return arr
