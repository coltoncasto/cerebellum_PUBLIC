addpath('/om/group/evlab/software/spm12');
files = dir('nii/*.nii');
data = readtable('../../gss/sessions/final_lang_atlas_sessions.csv');
coverage = readtable('../../gss/coverage/lana_cerebellar_coverage.csv');
assert(size(data,1)==size(coverage,1));
% n=3 with only 1 data file, n=49 missing>=50 cerebellar parcels
data = data([data.Runs>=2 & coverage.CerebellumVoxelsMissing<50],:); % n=754
atlas = zeros(91, 109, 91, size(data,1));
contrast='S-N'; % TO FILL
for idx=1:size(data,1)
    file_idx = find(arrayfun(@(x) contains(files(x).name,data.Session(idx)),1:length(files)));
    vol = spm_read_vols(spm_vol(['nii/' files(file_idx).name]));
    vals = vol(~(isnan(vol)|vol==0));
    thresh = prctile(vals, 90, 'all');
    atlas(:,:,:,idx) = (vol > thresh);
end
atlas = mean(atlas, 4);
info = niftiinfo('../LanA_GSS_SN_reduced_subjects_percentile-whole-brain_0.1_0.1/fROIs.nii');
info.Datatype = 'double';
niftiwrite(atlas,'atlas_reduced_subjects.nii',info,'version','NIfTI1');