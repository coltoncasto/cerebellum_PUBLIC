import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import *

def plot_audvis_and_spcorrs_together_averagedLH(result_versions_resp, experiments_resp, conditions_resp, colors_resp,
                                     result_versions_spcorr, experiments_spcorr, colors_spcorr, title):

    # produce separate plot for each GSS version
    for version_resp, version_spcorr in zip(result_versions_resp,result_versions_spcorr):
        print('\n'+version_resp+':'+'\n'+'-'*(len(version_resp)+1))

        # load in ROI labels and sample file to know the number of ROIs
        if 'standard' in version_resp:
            gss_version = 'parcels/language_standardParcels'
            label_filename = 'language_standardParcels_labels.txt'
        elif 'MDLOC' in version_resp:
            gss_version = 'parcels/MD_standardParcels'
            label_filename = 'MD_standardParcels_labels.txt'
        elif 'ToMLOC' in version_resp:
            gss_version = 'parcels/ToM_cerebellumParcels/ToM_GSS_bel_pho_'+'_'.join(version.split('_')[-3:])
            if 'whole-brain' in version_resp:
                label_filename = 'labels_cerebellum_only.txt'
            else:
                label_filename = 'labels.txt'
        else:
            gss_version = "".join(['LanA_GSS_SN_reduced_subjects_','_'.join(version_resp.split('_')[-3:])])
            if 'language' in version_resp:
                label_filename = 'labels.txt'
            elif 'symmetrical' in version_resp:
                label_filename = 'labels_cerebellum_only_symmetrical.txt'
            elif 'whole-brain' in version_resp:
                label_filename = 'labels_cerebellum_only.txt'
            else:
                label_filename = 'labels.txt'
        labels = pd.read_csv(os.path.join('../gss',gss_version,label_filename), sep=" ", header=None)
        
        spcorr_file = os.path.join('spcorr','spcorr_'+experiments_spcorr[0]+'_'+version_spcorr,'spm_bcc_data.EffectSize.csv')
        data = pd.read_csv(spcorr_file)

        
        for r,roi in enumerate(['AveragedLH']):
            print('ROI '+str(roi))

            # put all ROIs on same plot
            plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':True, 'top':False})
            plt.rc('font',**{'family':'sans-serif'})
            plt.rc('pdf', **{'fonttype':42})
            fig, ax = plt.subplots(figsize=(8,7),visible=True)
            sliding_x = 1
            xtickvalues = []
            xticklabels = []

            # --- AUD/VIS RESPONSES ---
            
            # show all experiments on the same plot
            for e,expt in enumerate(experiments_resp):
                if type(expt) is not list:
                    expt = [expt]
                curr_LOC = expt[0].split('_')[-1].split('LOC')[0]
                conds_curr = conditions_resp[e]
                colors_curr = colors_resp[e]
                
                # load toolbox output to plot
                mROI_file = os.path.join(version_resp.replace('*',expt[0]),'spm_ss_mROI_data.csv')
                print(mROI_file)
                data = pd.read_csv(mROI_file)
            
                conds_to_plot = np.zeros((len(pd.unique(data.Subject)),len(conds_curr)))
                n_scatter_points = np.zeros(len(conds_curr))
                for i,cond in enumerate(conds_curr): 
                    for sub_roi in ['LH_IFG','LH_IFGorb','LH_MFG','LH_PostTemp','LH_AntTemp']:
                        conds_to_plot[:,i] += data.loc[(data.ROI==sub_roi) & (data.Effect==cond)]['EffectSize']
                        n_scatter_points[i] = conds_to_plot.shape[0]-np.sum(np.isnan(conds_to_plot[:,i]))
                
                conds_to_plot = conds_to_plot/5 # average over regions    
                
                print('  Number of subjects (by condition): ',n_scatter_points)

                xtickvalues.append(np.arange(sliding_x,sliding_x+len(conds_curr)).tolist())
                xticklabels.append([conds_curr[c]+'('+str(int(n_scatter_points[c]))+')' for c in np.arange(0,len(conds_curr))])
                ax.bar(x=np.arange(sliding_x,sliding_x+len(conds_curr)),
                        height=np.nanmean(conds_to_plot,axis=0),
                        yerr=np.nanstd(conds_to_plot,axis=0)/np.sqrt(n_scatter_points),
                        edgecolor='k',
                        linewidth=2,
                        color=colors_curr,
                        capsize=12,
                        error_kw={'elinewidth' : 3, 'capthick' : 3}
                )
                sliding_x += len(conds_curr)

            xtickvalues = [item for expt in xtickvalues for item in expt] # flatten
            xticklabels = [item for expt in xticklabels for item in expt] # flatten
            
            plt.xticks(xtickvalues,xticklabels,rotation=45,fontsize=14,ha='right')
            ax.set_ylabel('% BOLD SIGNAL CHANGE',fontsize=18)
            # ax.set_ylim([0,2.7])

            
            # --- SPCORRS ---

            ax2 = ax.twinx()
            geometric_mean = []
            geometric_mean_all_data = pd.DataFrame()
            sliding_x_start_spcorr = sliding_x+1
            sliding_x += 1.5
            
            # show all experiments on the same plot
            for e,expt in enumerate(experiments_spcorr):
                
                # load toolbox output to plot
                if type(expt) is list:
                    data = pd.DataFrame()
                    for ex in expt:
                        spcorr_file = os.path.join('spcorr','spcorr_'+ex+'_'+version_spcorr,'spm_bcc_data.EffectSize.csv')
                        data_temp = pd.read_csv(spcorr_file)
                        data = pd.concat([data,data_temp])
                    data = data.groupby(['ROI','Subject']).agg({ 'Fisher transformed correlation coefficients' : ['mean']}).reset_index()
                    data.columns = data.columns.droplevel(1)
                else:
                    spcorr_file = os.path.join('spcorr','spcorr_'+experiments_spcorr[e]+'_'+version_spcorr,'spm_bcc_data.EffectSize.csv')
                    data = pd.read_csv(spcorr_file)
                    expt = [expt]
                
                spcorrs = np.zeros((len(pd.unique(data.Subject)),1))
                print(spcorrs.shape)
                for sub_roi in [1,2,3,4,5]:
                    spcorrs[:] += data.loc[(data.ROI==sub_roi)]['Fisher transformed correlation coefficients'].to_numpy().reshape(85,1)
                spcorrs = spcorrs/5 # average over regions
                print(len(spcorrs))
                print(spcorrs[:,0])
                print(expt)
                geometric_mean_all_data[expt[0]] = spcorrs[:,0]
                n_scatter_points = len(spcorrs)

                if type(experiments_spcorr[e]) is list:
                    hatch = '/'
                else:
                    hatch = None
                ax2.bar(x=sliding_x,
                        height=np.nanmean(spcorrs,axis=0),
                        yerr=np.nanstd(spcorrs,axis=0)/np.sqrt(n_scatter_points),
                        edgecolor='k',
                        linewidth=2,
                        color=colors_spcorr[e],
                        capsize=12,
                        hatch=hatch,
                        error_kw={'elinewidth' : 3, 'capthick' : 3}
                )

                if expt[0] in ['SN_SN','ID_ID']:
                        geometric_mean.append(np.nanmean(spcorrs,axis=0))

                if type(experiments_spcorr[e]) is list:
                    e_string = 'ID_SN'
                else:
                    e_string = expt[0]
                xtickvalues.append(sliding_x)
                xticklabels.append(e_string+' ('+str(len(spcorrs))+')')
                sliding_x = sliding_x + 1  


            # final plot params
            ax2.set_xticks(xtickvalues,xticklabels,rotation=45,fontsize=14,ha='right')
            # left,right = plt.xlim()
            ax2.plot([0,sliding_x],[0,0],'k')
            assert(len(geometric_mean)==2)
            geometric_to_plot = np.sqrt(geometric_mean[0]*geometric_mean[1])
            print(geometric_mean)
            print(geometric_to_plot)
            ax2.plot([sliding_x_start_spcorr,sliding_x-0.5],[geometric_to_plot,geometric_to_plot],'--',color='gainsboro')
            geometric_mean_all_data.to_csv('spcorr/spcorrs_summary_neocortex.csv',index=False)
            ax2.set_xlim([0,sliding_x])
            # ax2.set_ylim([0,1.2])
            ax2.set_ylabel('Fisher-transformed Correlation Coefficients',fontsize=18)
            plt.title('averagedLH'+'\n'+title,fontsize=20)
    
            # save         
            version_no_star = version_resp.replace('*_','')
            PLOTS_DIR = os.path.join('_plots',version_no_star)
            if not os.path.exists(PLOTS_DIR):
                os.makedirs(PLOTS_DIR)
            savename = os.path.join(PLOTS_DIR,"".join([version_no_star,'_averagedLH_',title,'_no_scatter.pdf']))
            plt.savefig(savename,format='pdf',bbox_inches='tight',pad_inches=0.5)
            plt.close()