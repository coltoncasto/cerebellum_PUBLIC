% taken from _from_ben/make_final_parcels.m

% load parcels
vol_parcels_orig = niftiread(niftiinfo('LanA_GSS_SN_reduced_subjects_percentile-whole-brain_0.1_0.1/fROIs_cerebellum_only.nii'));
vol_parcels = vol_parcels_orig;

vol_parcels(91:-1:46,:,:)=vol_parcels(1:1:46,:,:); % flip over
% change numbers in right hem
vol_parcels(vol_parcels==1)=5; % left CrusII
vol_parcels(vol_parcels==2)=6; % left CrusI dentate
vol_parcels(vol_parcels==3)=7; % left lobule VIIIb
vol_parcels(vol_parcels==4)=8; % left CrusI
% replace left hem to have original numbering
vol_parcels(1:46,:,:) = vol_parcels_orig(1:46,:,:);

% save file
info = niftiinfo('LanA_GSS_SN_reduced_subjects_percentile-whole-brain_0.1_0.1/fROIs_cerebellum_only.nii');
niftiwrite(vol_parcels,'LanA_GSS_SN_reduced_subjects_percentile-whole-brain_0.1_0.1/fROIs_cerebellum_only_symmetrical.nii',info);