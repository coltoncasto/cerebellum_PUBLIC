import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
from helpers import *

def plot_experiments_together(result_versions,
                              experiments,
                              conditions,
                              colors,
                              title,
                              no_outliers=True,
                              set_ylim=False,
                              ylim=None,
                              include_scatter=True):

    # initial variables
    n_conds = np.sum(np.array([len(cond) for cond in conditions]))
    if no_outliers:
        suffix = '_no_outliers.csv'
        suffix_fig = '_no_outliers'
    else:
        suffix = '.csv'
        suffix_fig = ''
    if set_ylim:
        suffix_fig2 = '_trimmedYaxis'
    else:
        suffix_fig2 = ''
    if include_scatter:
        height = 15
        suffix_scat = ''
    else:
        height = 8 # 10 for nonling 
        suffix_scat = '_no_scatter'
    
    # produce separate plot for each mROI+GSS version
    for version in result_versions:
        print('\n'+version+':'+'\n'+'-'*(len(version)+1))

        # load in ROI labels and sample file to know the number of ROIs
        if type(experiments[0]) is list:
            expt1 = experiments[0][0]
        else:
            expt1 = experiments[0]
        if expt1=='langloc':
            version_use = version.replace('LOC','EFFECT')
        else:
            version_use = version
        mROI_file = glob.glob(os.path.join(expt1,'data',version_use+suffix))[0]
        data = pd.read_csv(mROI_file)
        
        # load in ROI labels
        if 'standard' in version:
            gss_version = 'parcels/language_standardParcels'
            label_filename = 'language_standardParcels_labels.txt'
        else:
            gss_version = "".join(['LanA_GSS_SN_reduced_subjects_','_'.join(version.split('_')[-3:])])
            if 'language' in version:
                label_filename = 'labels.txt'
            elif 'symmetrical' in version:
                label_filename = 'labels_cerebellum_only_symmetrical.txt'
            elif 'whole-brain' in version:
                label_filename = 'labels_cerebellum_only.txt'
            else:
                label_filename = 'labels.txt'
        labels = pd.read_csv(os.path.join('../gss',gss_version,label_filename), sep=" ", header=None)

        n_bars = np.sum(np.array([len(item) for item in conditions]))
        source_data = []
        source_data_subs = []
        source_data_mean = np.zeros((len(pd.unique(data.ROI)),n_bars))
        source_data_sem = np.zeros((len(pd.unique(data.ROI)),n_bars))
        source_data_n = np.zeros((len(pd.unique(data.ROI)),n_bars))
        if set_ylim:
            source_data_cutoff = np.zeros((len(pd.unique(data.ROI)),n_bars))
        expt_str_list = []

        if set_ylim:
            assert(len(np.sort(pd.unique(data.ROI)))==len(ylim))
        
        # produce separate plot for each ROI
        for r,roi in enumerate(np.sort(pd.unique(data.ROI))):
            print('ROI '+str(roi)+': '+labels.iloc[roi-1,0])
            source_data_curr_roi = []
            source_data_curr_roi_subs = []

            # plot
            plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})
            plt.rc('font',**{'family':'sans-serif'})
            plt.rc('pdf', **{'fonttype':42})
            fig, ax = plt.subplots(figsize=(n_conds+len(experiments),height),visible=True)
            sliding_x = 0.5
            xtickvalues = []
            xticklabels = []
            col = 0
            
            # show all experiments on the same plot
            for e,expt in enumerate(experiments):
                if type(expt) is not list:
                    expt = [expt]
                conds_curr = conditions[e]
                colors_curr = colors[e]
                sliding_x_start = sliding_x
                
                # load toolbox output to plot
                data = pd.DataFrame()
                for ex in expt:
                    if ex=='langloc':
                        version_use = version.replace('LOC','EFFECT')
                    else:
                        version_use = version
                    mROI_file = glob.glob(os.path.join(ex,'data',version_use+suffix))[0]
                    print(mROI_file)
                    data_curr = pd.read_csv(mROI_file)
                    data = pd.concat([data,data_curr], axis=0)

                # change experiment name for ParamNew_2013_2015_ips230_240
                if expt[0]=='ParamNew_2013_2015_ips230_240':
                    expt_use = 'ParamNew'
                else:
                    expt_use = expt

                
                nSubs = len(data.loc[(data.ROI==roi)][conds_curr[0]])
                nConds = len(conds_curr)
                conds_to_plot = np.zeros((nSubs,nConds))
                conds_to_plot_xs = np.zeros((nSubs,nConds))
                conds_to_plot_colors = []
                n_scatter_points = np.zeros(nConds)
                for i,cond in enumerate(conds_curr): 
                    conds_to_plot[:,i] = data.loc[(data.ROI==roi)][cond]
                    conds_to_plot_xs[:,i] = generate_jittered_points(sliding_x, conds_to_plot_xs.shape[0])
                    sliding_x = sliding_x + 1
                    conds_to_plot_colors.append([colors_curr[i]]*conds_to_plot.shape[0])
                    n_scatter_points[i] = conds_to_plot.shape[0]-np.sum(np.isnan(conds_to_plot[:,i]))

                    # store data 
                    source_data_curr_roi.append(conds_to_plot[:,i])
                    source_data_curr_roi_subs.append(data.loc[(data.ROI==roi)]['SessionCritical'])
                    source_data_mean[r,col] = np.nanmean(conds_to_plot[:,i],axis=0)
                    source_data_sem[r,col] = np.nanstd(conds_to_plot[:,i],axis=0)/np.sqrt(n_scatter_points[i])
                    source_data_n[r,col] = n_scatter_points[i]
                    if set_ylim:
                        source_data_cutoff[r,col] = np.sum((conds_to_plot[:,i]<ylim[r][0]) | (conds_to_plot[:,i]>ylim[r][1]))
                    col = col+1
                    if r==0: # only construct list on first pass
                        expt_str_list.append('_'.join(expt)+'_'+cond)
                
                conds_to_plot_colors = [item for cond in conds_to_plot_colors for item in cond] # flatten 
                print('  Number of subjects (by condition): ',n_scatter_points)

                if expt[0] not in ['MDloc_atlas','DigSpan','MSIT','vMSIT','Math_2015onward_AliceLocalizer','Music_2009']:
                    sliding_x = sliding_x + 1.5 # slide an extra 2 spaces between experiments

                # plot
                xtickvalues.append(np.arange(sliding_x_start,sliding_x_start+len(conds_curr)).tolist())
                xticklabels.append([conds_curr[c]+'('+str(int(n_scatter_points[c]))+')' for c in np.arange(0,len(conds_curr))])
                ax.bar(x=np.arange(sliding_x_start,sliding_x_start+len(conds_curr)),
                        height=np.nanmean(conds_to_plot,axis=0),
                        yerr=np.nanstd(conds_to_plot,axis=0)/np.sqrt(n_scatter_points),
                        edgecolor='k',
                        linewidth=2,
                        color=colors_curr,
                        capsize=15,
                        error_kw={'elinewidth' : 3, 'capthick' : 3}
                )

                if include_scatter:
                    if np.any(n_scatter_points>60):
                        alpha = 0.1
                    else:
                        alpha = 0.3
                    plt.scatter(conds_to_plot_xs.flatten('F'),
                                conds_to_plot.flatten('F'),
                                s=np.array([60]*conds_to_plot.shape[0]*len(conds_curr)),
                                c=conds_to_plot_colors,
                                edgecolors='k',
                                alpha=alpha
                    )

                if expt[0]=='langloc':
                    langloc_ys = np.nanmean(conds_to_plot,axis=0)
    
            source_data.append(source_data_curr_roi)
            source_data_subs.append(source_data_curr_roi_subs)

            if set_ylim:
                plt.ylim([ylim[r][0],ylim[r][1]])
            bottom,top = plt.ylim()
            y_vals = np.arange(top,bottom,-1*(top-bottom)/40)
            for e,expt in enumerate(experiments):
                if type(expt) is list:
                    expt = '+'.join(expt)
                ax.annotate(str(e+1)+') '+expt, xy=(sliding_x-1.5,y_vals[e]), xycoords='data', textcoords=('data','offset points'),
                            xytext=(sliding_x-1.5,y_vals[e]), ha='right', va='top', fontsize=12)

            xtickvalues = [item for expt in xtickvalues for item in expt] # flatten
            xticklabels = [item for expt in xticklabels for item in expt] # flatten
            plt.xticks(xtickvalues,xticklabels,rotation=45,fontsize=14,ha='right')
            left,right = plt.xlim()
            plt.plot([left,right],[0,0],'k')
            if title=='NONLING_FIGURE':
                plt.plot([0.5,sliding_x-1.5],[langloc_ys[0],langloc_ys[0]],'--',c=[0.15,0.15,0.15]) # sentences dashed line
                plt.plot([1.5,sliding_x-1.5],[langloc_ys[1],langloc_ys[1]],'--',c=[0.75,0.75,0.75]) # nonword dashed line
            plt.xlim([left,right])
            plt.yticks(fontsize=16)
            plt.ylabel('% BOLD SIGNAL CHANGE',fontsize=18)
            plt.title('ROI '+str(roi)+': '+labels.iloc[roi-1,0]+'\n'+title,fontsize=20)
    
            # save         
            if 'language_standard' in version:
                version_no_star = version[0:9]+'language_standard';
            elif 'language' in version:
                version_no_star = version[0:9]+'language_GSS'+version.split('GSS')[-1]
            elif 'symmetrical' in version:
                version_no_star = version[0:9]+'symmetrical_GSS'+version.split('GSS')[-1]
            else:
                version_no_star = version[0:9]+'GSS'+version.split('GSS')[-1]
            # save plot
            PLOTS_DIR = os.path.join('_plots',title,version_no_star)
            if not os.path.exists(PLOTS_DIR):
                os.makedirs(PLOTS_DIR)
            savename = os.path.join(PLOTS_DIR,"".join([version_no_star,'_ROI_',str(roi),'_',labels.iloc[roi-1,0],'_',title,suffix_fig,suffix_scat,suffix_fig2,'.pdf'])) # _uniform_colors
            plt.savefig(savename,format='pdf',bbox_inches='tight',pad_inches=0.5)
            plt.close()
            
        # save data
        DATA_DIR = os.path.join('_data',title)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        labels_for_df = [labels.iloc[roi-1,0] for roi in np.sort(pd.unique(data.ROI))]

        # format source data
        multi = pd.MultiIndex.from_product([labels_for_df,expt_str_list],names=['ROIs','Conditions'])
        source_data_flattened = [cond for roi in source_data for cond in roi] # flatten roi*conds
        max_n = max([len(x) for x in source_data_flattened])
        source_data_padded = [np.append(x,np.array([np.nan]*(max_n-len(x)))) for x in source_data_flattened]
        cols = ['s'+str(i) for i in np.arange(1,max_n+1)]
        source_data_df = pd.DataFrame(source_data_padded,index=multi,columns=cols)
        savename = os.path.join(DATA_DIR,"".join([version_no_star,'_',title,suffix_fig,'_source_data.csv']))
        source_data_df.to_csv(savename)

        # format source data subs
        source_data_subs_flattened = [cond for roi in source_data_subs for cond in roi] # flatten roi*conds
        max_n = max([len(x) for x in source_data_subs_flattened])
        source_data_subs_padded = [np.append(x,np.array([np.nan]*(max_n-len(x)))) for x in source_data_subs_flattened]
        cols = ['s'+str(i) for i in np.arange(1,max_n+1)]
        source_data_subs_df = pd.DataFrame(source_data_subs_padded,index=multi,columns=cols)
        savename = os.path.join(DATA_DIR,"".join([version_no_star,'_',title,suffix_fig,'_source_data_subs.csv']))
        source_data_subs_df.to_csv(savename)

        # save condensed source data
        savename = os.path.join(DATA_DIR,"".join([version_no_star,'_',title,suffix_fig,'_source_data_mean.csv']))
        source_data_mean_df = pd.DataFrame(source_data_mean,index=labels_for_df,columns=expt_str_list)
        source_data_mean_df.to_csv(savename)
        savename = os.path.join(DATA_DIR,"".join([version_no_star,'_',title,suffix_fig,'_source_data_sem.csv']))
        source_data_sem_df = pd.DataFrame(source_data_sem,index=labels_for_df,columns=expt_str_list)
        source_data_sem_df.to_csv(savename)
        savename = os.path.join(DATA_DIR,"".join([version_no_star,'_',title,suffix_fig,'_source_data_n.csv']))
        source_data_n_df = pd.DataFrame(source_data_n,index=labels_for_df,columns=expt_str_list)
        source_data_n_df.to_csv(savename)
        if set_ylim:
            savename = os.path.join(DATA_DIR,"".join([version_no_star,'_',title,suffix_fig,'_source_data_cutoff.csv']))
            source_data_cutoff_df = pd.DataFrame(source_data_cutoff,index=labels_for_df,columns=expt_str_list)
            source_data_cutoff_df.to_csv(savename)
